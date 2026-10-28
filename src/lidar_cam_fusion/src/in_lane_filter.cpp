#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "vision_interfaces/msg/detection_array.hpp"

using det = vision_interfaces::msg::DetectionArray;
using cl = sensor_msgs::msg::PointCloud2;

class InLaneFilter : public rclcpp::Node {
    public:
    InLaneFilter() : Node("in_lane_filter"), tf_buffer(this->get_clock()), tf_listener(tf_buffer){

        //parameters
        std::string in_topic = declare_parameter<std::string>("input_topic", "detections_depth");
        std::string out_topic = declare_parameter<std::string>("output_topic", "detections_lane");
        std::string white_topic = declare_parameter<std::string>("white_lane_topic", "pc_whitelane");
        std::string yellow_topic = declare_parameter<std::string>("yellow_lane_topic", "pc_yellowlane");
        ground_frame = declare_parameter<std::string>("ground_frame", "link_base");
        max_lane_width = declare_parameter<double>("max_lane_width", 5.0);
        min_lane_width = declare_parameter<double>("min_lane_width", 1.0);
        lane_width = declare_parameter<double>("lane_width", 3.8);
        lane_margin = declare_parameter<double>("lane_margin", 0.0);
        lane_max_age = declare_parameter<double>("lane_max_age", 0.5);
        in_lane_unknown = declare_parameter<bool>("in_lane_unknown", false);
        //lane curve fitting / extrapolation
        bin_size = declare_parameter<double>("bin_size", 1.0);                   
        min_bin_points = declare_parameter<int>("min_bin_points", 2);
        min_fit_span = declare_parameter<double>("min_fit_span", 2.0);           
        max_extrapolation = declare_parameter<double>("max_extrapolation", 15.0); 
        max_slope = declare_parameter<double>("max_slope", 0.6);                 
        track_tol = declare_parameter<double>("track_tol", 0.3);                 
        require_bracket = declare_parameter<bool>("require_bracket", true);     

        //publishers and subscribers
        white_sub = this->create_subscription<cl>(white_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::white_callback, this, std::placeholders::_1));
        yellow_sub = this->create_subscription<cl>(yellow_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::yellow_callback, this, std::placeholders::_1));
        detection_subscription = this->create_subscription<det>(in_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::detection_callback, this, std::placeholders::_1));
        publisher = this->create_publisher<det>(out_topic,10);
    }

    private:

    struct P2 {
        float x,y;
    };

    // Boundary of the bot's own lane as y = a + b*x + c*x^2 
   struct Curve {
        bool ok = false;
        double x_min = 0, x_max = 0;
        double a = 0, b = 0, c = 0;
        double y_near = 0;   
    };

    struct LaneCloud {
        Curve curve;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        bool valid = false;
    };

    enum class Lane {IN,OUT,UNKNOWN};
    std::string ground_frame;
    double max_lane_width, min_lane_width, lane_margin, lane_max_age, lane_width;
    double bin_size, min_fit_span, max_extrapolation, max_slope;
    double track_tol;
    int min_bin_points;
    bool in_lane_unknown, require_bracket;
    LaneCloud white, yellow;

    rclcpp::Subscription<cl>::SharedPtr white_sub, yellow_sub;
    rclcpp::Subscription<det>::SharedPtr detection_subscription;
    rclcpp::Publisher<det>::SharedPtr publisher;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    bool lookup(const std::string &target, const std::string &source,
                const builtin_interfaces::msg::Time &stamp, geometry_msgs::msg::TransformStamped &result) {
        try {
            result = tf_buffer.lookupTransform(target, source, tf2_ros::fromMsg(stamp), tf2::durationFromSec(0.05));
            return true;
        } catch (const tf2::TransformException &) {
            try {
                result = tf_buffer.lookupTransform(target, source, tf2::TimePointZero);
                return true;
            } catch (const tf2::TransformException &e) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Transform failed: %s", e.what());
                return false;
            }
        }
    }

    void white_callback(const cl::SharedPtr msg) {
        update(white, msg);
    }

    void yellow_callback(const cl::SharedPtr msg) {
        update(yellow, msg);
    }

    Curve fit_curve(const std::vector<P2> &pts) const {
        Curve cv;
        std::map<int, std::vector<P2>> bins;
        for (const auto &p : pts) {
            if (p.x < 0.0f || std::fabs(p.y) > max_lane_width) continue;   
            bins[static_cast<int>(std::floor(p.x / bin_size))].push_back(p);
        }

        std::vector<P2> s;
        bool seeded = false;
        double prev_x = 0, prev_y = 0;
        for (const auto &kv : bins) {
            const auto &v = kv.second;
            if (static_cast<int>(v.size()) < min_bin_points) continue;
            const P2 *best = nullptr;
            double best_d = std::numeric_limits<double>::infinity();
            for (const auto &p : v) {
                double d = seeded ? std::fabs(p.y - prev_y) : std::fabs(p.y);
                if (d < best_d) { best_d = d; best = &p; }
            }
            if (!best) continue;
            if (!seeded) {
                if (best_d > lane_width) continue;
            } else {
                double tol = track_tol + max_slope * (best->x - prev_x);
                if (best_d > tol) continue;
            }
            s.push_back(*best);
            prev_x = best->x;
            prev_y = best->y;
            seeded = true;
        }
        if (s.size() < 2) return cv;

        const double span = s.back().x - s.front().x;
        if (span < min_fit_span) return cv;

        const int deg = (s.size() >= 4 && span >= 3.0) ? 2 : 1;
        Eigen::MatrixXd A(s.size(), deg + 1);
        Eigen::VectorXd yv(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            A(i, 0) = 1.0;
            A(i, 1) = s[i].x;
            if (deg == 2) A(i, 2) = static_cast<double>(s[i].x) * s[i].x;
            yv(i) = s[i].y;
        }
        Eigen::VectorXd c = A.colPivHouseholderQr().solve(yv);
        cv.a = c(0);
        cv.b = c(1);
        cv.c = (deg == 2) ? c(2) : 0.0;
        cv.x_min = s.front().x;
        cv.x_max = s.back().x;
        cv.y_near = cv.a + cv.b * cv.x_min + cv.c * cv.x_min * cv.x_min;
        cv.ok = true;
        return cv;
    }

    void update(LaneCloud &lane, const cl::ConstSharedPtr &cloud) {
        geometry_msgs::msg::TransformStamped tf;
        //Apply transformation
        try{
            tf = tf_buffer.lookupTransform(ground_frame, cloud->header.frame_id, tf2::TimePointZero);
        }
        catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Lookup transform failed");
            return;
        }

        cl tc;
        tf2::doTransform(*cloud, tc, tf);

        //Extract points
        std::vector<P2> pts;
        pts.reserve(static_cast<size_t>(tc.width) * tc.height);
        sensor_msgs::PointCloud2ConstIterator<float> ix(tc,"x");
        sensor_msgs::PointCloud2ConstIterator<float> iy(tc,"y");
        sensor_msgs::PointCloud2ConstIterator<float> iz(tc,"z");

        for (; ix != ix.end(); ++ix, ++iy, ++iz) {
            if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz))
                continue;
            pts.push_back({*ix, *iy});
        }

        lane.curve = fit_curve(pts);
        lane.stamp = rclcpp::Time(cloud->header.stamp);
        lane.valid = lane.curve.ok;
    }

    //Check whether data is fresh
    bool fresh(const LaneCloud &lane, const rclcpp::Time &now) const {
        return lane.valid && std::fabs((now - lane.stamp).seconds()) <= lane_max_age;
    }

    bool eval(const Curve &cv, double x, double &y) const {
        if (!cv.ok || x < cv.x_min - max_extrapolation || x > cv.x_max + max_extrapolation)
            return false;
        const double xe = std::clamp(x, cv.x_min, cv.x_max);
        y = cv.a + cv.b * xe + cv.c * xe * xe;
        if (x != xe) {
            const double slope = std::clamp(cv.b + 2.0 * cv.c * xe, -max_slope, max_slope);
            y += slope * (x - xe);
        }
        return true;
    }

    //Main function to check whether in lane
    Lane check(double y, bool has_white, double y_white, bool has_yellow, double y_yellow) const {
        double hi=0, lo=0;
        if (has_white && has_yellow) {
            // the bot must be between its own white and yellow line 
            if (require_bracket && white.curve.y_near * yellow.curve.y_near >= 0.0)
                return Lane::UNKNOWN;
            lo = std::min(y_white, y_yellow);
            hi = std::max(y_white, y_yellow);
            if(hi-lo > max_lane_width || hi-lo < min_lane_width)
                return Lane::UNKNOWN;
        }
        else if(has_white || has_yellow) {
            const Curve &c = has_yellow ? yellow.curve : white.curve;
            double y_line = has_yellow ? y_yellow : y_white;
            bool left = c.y_near > 0;     
            lo = left ? y_line - lane_width : y_line;
            hi = left ? y_line : y_line + lane_width;
        }
        else
            return Lane::UNKNOWN;

        if (y >= lo - lane_margin && y <= hi + lane_margin)
            return Lane::IN;
        else
            return Lane::OUT;
    }

    void detection_callback(const det::ConstSharedPtr msg) {
        det output = *msg;
        const rclcpp::Time now(msg->header.stamp);
        const bool white_fresh = fresh(white, now);
        const bool yellow_fresh = fresh(yellow, now);
        const bool lanes_ok = white_fresh || yellow_fresh;
        if(lanes_ok == false) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Lane cloud missing");
        }

        //Transform from detection frame to ground frame
        bool have_tf = false;
        geometry_msgs::msg::TransformStamped g;
        if(lanes_ok && lookup(ground_frame, msg->header.frame_id, msg->header.stamp, g))
            have_tf = true;

        for (auto &item : output.detections) {
            if(item.depth_valid == false) {
                item.is_in_lane = false;
                continue;
            }
            Lane r = Lane::UNKNOWN;
            if(have_tf) {
                geometry_msgs::msg::PointStamped source_point;
                geometry_msgs::msg::PointStamped target_point;
                source_point.header = msg->header;
                source_point.point = item.position;
                tf2::doTransform(source_point, target_point, g);

                double y_white = 0, y_yellow = 0;
                bool has_white = white_fresh && eval(white.curve, target_point.point.x, y_white);
                bool has_yellow = yellow_fresh && eval(yellow.curve, target_point.point.x, y_yellow);
                r = check(target_point.point.y, has_white, y_white, has_yellow, y_yellow);

                //debug: result 0 = IN, 1 = OUT, 2 = UNKNOWN. xmax = farthest point of that lane cloud.
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "obj x=%.2f y=%.2f | white=%d y=%.2f near=%.2f xmax=%.1f | yellow=%d y=%.2f near=%.2f xmax=%.1f -> %d",
                    target_point.point.x, target_point.point.y,
                    has_white, y_white, white.curve.y_near, white.curve.x_max,
                    has_yellow, y_yellow, yellow.curve.y_near, yellow.curve.x_max,
                    static_cast<int>(r));
            }
            item.is_in_lane = (r == Lane::IN) || (r == Lane::UNKNOWN && in_lane_unknown);
        }
        publisher->publish(output);
    }
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InLaneFilter>());
    rclcpp::shutdown();

    return 0;
}