/*

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
        lane_width = declare_parameter<double>("lane_width", 3.5);
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

*/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
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
#include "visualization_msgs/msg/marker_array.hpp"

using det = vision_interfaces::msg::DetectionArray;
using cl = sensor_msgs::msg::PointCloud2;
using clk = std::chrono::steady_clock;

// ------------------------------------------------------------------------------------------------
// DEBUG BUILD + MARKER ARRAY VISUALIZATION
// ------------------------------------------------------------------------------------------------

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
        lane_width = declare_parameter<double>("lane_width", 3.5);
        lane_margin = declare_parameter<double>("lane_margin", 0.0);
        lane_max_age = declare_parameter<double>("lane_max_age", 0.5);
        in_lane_unknown = declare_parameter<bool>("in_lane_unknown", false);
        bin_size = declare_parameter<double>("bin_size", 1.0);
        min_bin_points = declare_parameter<int>("min_bin_points", 2);
        min_fit_span = declare_parameter<double>("min_fit_span", 2.0);
        max_extrapolation = declare_parameter<double>("max_extrapolation", 15.0);
        max_slope = declare_parameter<double>("max_slope", 0.6);
        track_tol = declare_parameter<double>("track_tol", 0.3);
        require_bracket = declare_parameter<bool>("require_bracket", true);
        // debug
        debug = declare_parameter<bool>("debug", true);
        debug_period = declare_parameter<double>("debug_period", 0.5);
        debug_bins = declare_parameter<bool>("debug_bins", true);

        //publishers and subscribers
        white_sub = this->create_subscription<cl>(white_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::white_callback, this, std::placeholders::_1));
        yellow_sub = this->create_subscription<cl>(yellow_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::yellow_callback, this, std::placeholders::_1));
        detection_subscription = this->create_subscription<det>(in_topic, rclcpp::SensorDataQoS(), std::bind(&InLaneFilter::detection_callback, this, std::placeholders::_1));
        publisher = this->create_publisher<det>(out_topic, 10);
        marker_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("in_lane_filter_markers", 10);

        heartbeat_timer = this->create_wall_timer(std::chrono::seconds(1), std::bind(&InLaneFilter::heartbeat, this));

        RCLCPP_INFO(get_logger(), "[INIT] in='%s' out='%s' white='%s' yellow='%s' ground_frame='%s'",
            in_topic.c_str(), out_topic.c_str(), white_topic.c_str(), yellow_topic.c_str(), ground_frame.c_str());
        RCLCPP_INFO(get_logger(), "[INIT] lane: max_lane_width=%.2f min_lane_width=%.2f lane_width=%.2f lane_margin=%.2f lane_max_age=%.2f in_lane_unknown=%d require_bracket=%d",
            max_lane_width, min_lane_width, lane_width, lane_margin, lane_max_age, in_lane_unknown, require_bracket);
        RCLCPP_INFO(get_logger(), "[INIT] fit: bin_size=%.2f min_bin_points=%d min_fit_span=%.2f track_tol=%.2f max_extrapolation=%.1f max_slope=%.2f",
            bin_size, min_bin_points, min_fit_span, track_tol, max_extrapolation, max_slope);
        RCLCPP_INFO(get_logger(), "[INIT] debug=%d debug_period=%.2f debug_bins=%d", debug, debug_period, debug_bins);
    }

    private:

    struct P2 {
        float x,y;
    };

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

    struct Stats {
        unsigned long long n = 0, prev = 0, fit_fail = 0, flips = 0;
        clk::time_point last{};
        bool any = false;
    };

    enum class Lane {IN,OUT,UNKNOWN};
    std::string ground_frame;
    double max_lane_width, min_lane_width, lane_margin, lane_max_age, lane_width;
    double bin_size, min_fit_span, max_extrapolation, max_slope;
    double track_tol;
    int min_bin_points;
    bool in_lane_unknown, require_bracket;
    LaneCloud white, yellow;

    bool debug = true, debug_bins = true;
    double debug_period = 0.5;
    clk::time_point dbg_white_t{}, dbg_yellow_t{}, dbg_det_t{};
    Stats white_stats, yellow_stats, det_stats;
    unsigned long long published = 0, published_prev = 0;
    rclcpp::TimerBase::SharedPtr heartbeat_timer;

    rclcpp::Subscription<cl>::SharedPtr white_sub, yellow_sub;
    rclcpp::Subscription<det>::SharedPtr detection_subscription;
    rclcpp::Publisher<det>::SharedPtr publisher;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    bool due(clk::time_point &last) {
        if (!debug) return false;
        const auto t = clk::now();
        if (debug_period > 0.0 && std::chrono::duration<double>(t - last).count() < debug_period) return false;
        last = t;
        return true;
    }

    static const char *lane_name(Lane r) {
        switch (r) {
            case Lane::IN: return "IN";
            case Lane::OUT: return "OUT";
            default: return "UNKNOWN";
        }
    }

    void heartbeat() {
        if (!debug) return;
        const auto t = clk::now();
        auto ago = [&](const Stats &s) { return s.any ? std::chrono::duration<double>(t - s.last).count() : -1.0; };
        RCLCPP_INFO(get_logger(),
            "[HEARTBEAT] msgs in last 1s: white=%llu yellow=%llu det=%llu published=%llu | seconds since last (-1 = never): white=%.2f yellow=%.2f det=%.2f",
            white_stats.n - white_stats.prev, yellow_stats.n - yellow_stats.prev, det_stats.n - det_stats.prev,
            published - published_prev, ago(white_stats), ago(yellow_stats), ago(det_stats));
        RCLCPP_INFO(get_logger(),
            "[HEARTBEAT] lane valid now: white=%d yellow=%d | total fit failures: white=%llu yellow=%llu | valid->invalid flips: white=%llu yellow=%llu",
            white.valid, yellow.valid, white_stats.fit_fail, yellow_stats.fit_fail, white_stats.flips, yellow_stats.flips);
        white_stats.prev = white_stats.n;
        yellow_stats.prev = yellow_stats.n;
        det_stats.prev = det_stats.n;
        published_prev = published;
    }

    void log_lane(const char *name, const LaneCloud &lane, const rclcpp::Time &now, bool is_fresh) const {
        if (lane.stamp.nanoseconds() == 0) {
            RCLCPP_INFO(get_logger(), "[DET] lane %s: NEVER RECEIVED a cloud (fresh=0)", name);
            return;
        }
        const double age = (now - lane.stamp).seconds();
        RCLCPP_INFO(get_logger(),
            "[DET] lane %s: valid=%d fresh=%d | lane stamp=%.3f age vs detection=%.3f s (limit lane_max_age=%.2f)",
            name, lane.valid, is_fresh, lane.stamp.seconds(), age, lane_max_age);
        if (lane.curve.ok) {
            RCLCPP_INFO(get_logger(),
                "[DET] lane %s: curve y=%.3f%+.3f*x%+.4f*x^2 on x[%.2f, %.2f], y_near=%.2f (%s of the bot)",
                name, lane.curve.a, lane.curve.b, lane.curve.c, lane.curve.x_min, lane.curve.x_max,
                lane.curve.y_near, lane.curve.y_near > 0 ? "LEFT" : "RIGHT");
        } else {
            RCLCPP_INFO(get_logger(), "[DET] lane %s: NO CURVE (last fit failed or no usable points)", name);
        }
    }

    std::string eval_reason(const Curve &cv, double x) const {
        char buf[260];
        if (!cv.ok) return "curve not valid";
        if (x < cv.x_min - max_extrapolation) {
            std::snprintf(buf, sizeof(buf), "REJECT x=%.2f is %.2f m before curve start %.2f (max_extrapolation=%.1f)",
                          x, cv.x_min - x, cv.x_min, max_extrapolation);
            return buf;
        }
        if (x > cv.x_max + max_extrapolation) {
            std::snprintf(buf, sizeof(buf), "REJECT x=%.2f is %.2f m beyond curve end %.2f (max_extrapolation=%.1f)",
                          x, x - cv.x_max, cv.x_max, max_extrapolation);
            return buf;
        }
        if (x < cv.x_min || x > cv.x_max) {
            const double xe = std::clamp(x, cv.x_min, cv.x_max);
            const double raw = cv.b + 2.0 * cv.c * xe;
            const double clamped = std::clamp(raw, -max_slope, max_slope);
            std::snprintf(buf, sizeof(buf), "EXTRAPOLATING %.2f m outside [%.2f, %.2f], slope raw=%.3f used=%.3f",
                          x < cv.x_min ? cv.x_min - x : x - cv.x_max, cv.x_min, cv.x_max, raw, clamped);
            return buf;
        }
        std::snprintf(buf, sizeof(buf), "inside fitted range [%.2f, %.2f]", cv.x_min, cv.x_max);
        return buf;
    }

    bool lookup(const std::string &target, const std::string &source,
                const builtin_interfaces::msg::Time &stamp, geometry_msgs::msg::TransformStamped &result,
                bool *used_fallback = nullptr) {
        if (used_fallback) *used_fallback = false;
        try {
            result = tf_buffer.lookupTransform(target, source, tf2_ros::fromMsg(stamp), tf2::durationFromSec(0.05));
            return true;
        } catch (const tf2::TransformException &e1) {
            try {
                result = tf_buffer.lookupTransform(target, source, tf2::TimePointZero);
                if (used_fallback) *used_fallback = true;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[TF] lookup %s<-%s at stamp failed (%s), used LATEST transform instead", target.c_str(), source.c_str(), e1.what());
                return true;
            } catch (const tf2::TransformException &e) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "[TF] Transform failed %s<-%s: %s", target.c_str(), source.c_str(), e.what());
                return false;
            }
        }
    }

    void white_callback(const cl::SharedPtr msg) {
        update(white, msg, "white", white_stats, dbg_white_t);
    }

    void yellow_callback(const cl::SharedPtr msg) {
        update(yellow, msg, "yellow", yellow_stats, dbg_yellow_t);
    }

    Curve fit_curve(const std::vector<P2> &pts, const char *name, bool dbg) const {
        Curve cv;
        std::map<int, std::vector<P2>> bins;
        size_t n_behind = 0, n_side = 0, n_kept = 0;
        for (const auto &p : pts) {
            if (p.x < 0.0f || std::fabs(p.y) > max_lane_width) {
                if (p.x < 0.0f) n_behind++; else n_side++;
                continue;
            }
            bins[static_cast<int>(std::floor(p.x / bin_size))].push_back(p);
            n_kept++;
        }
        if (dbg) {
            RCLCPP_INFO(get_logger(),
                "[FIT %s] step1 filter: input=%zu | dropped behind bot (x<0)=%zu | dropped too far sideways (|y|>%.1f)=%zu | kept=%zu | bins=%zu",
                name, pts.size(), n_behind, max_lane_width, n_side, n_kept, bins.size());
        }

        std::vector<P2> s;
        bool seeded = false;
        double prev_x = 0, prev_y = 0;
        for (const auto &kv : bins) {
            const auto &v = kv.second;
            const double bx0 = kv.first * bin_size;
            if (static_cast<int>(v.size()) < min_bin_points) {
                if (dbg && debug_bins)
                    RCLCPP_INFO(get_logger(), "[FIT %s] bin %3d x[%5.1f,%5.1f) n=%2zu SKIP: fewer than min_bin_points=%d",
                        name, kv.first, bx0, bx0 + bin_size, v.size(), min_bin_points);
                continue;
            }
            const P2 *best = nullptr;
            double best_d = std::numeric_limits<double>::infinity();
            for (const auto &p : v) {
                double d = seeded ? std::fabs(p.y - prev_y) : std::fabs(p.y);
                if (d < best_d) { best_d = d; best = &p; }
            }
            if (!best) continue;
            if (!seeded) {
                if (best_d > lane_width) {
                    if (dbg && debug_bins)
                        RCLCPP_INFO(get_logger(), "[FIT %s] bin %3d x[%5.1f,%5.1f) n=%2zu SEED REJECTED: nearest to centre line is (%.2f,%.2f), |y|=%.2f > lane_width=%.2f",
                            name, kv.first, bx0, bx0 + bin_size, v.size(), best->x, best->y, best_d, lane_width);
                    continue;
                }
            } else {
                double tol = track_tol + max_slope * (best->x - prev_x);
                if (best_d > tol) {
                    if (dbg && debug_bins)
                        RCLCPP_INFO(get_logger(), "[FIT %s] bin %3d x[%5.1f,%5.1f) n=%2zu REJECT: closest to previous (%.2f,%.2f) is (%.2f,%.2f), dy=%.2f > tol=%.2f",
                            name, kv.first, bx0, bx0 + bin_size, v.size(), prev_x, prev_y, best->x, best->y, best_d, tol);
                    continue;
                }
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

    void update(LaneCloud &lane, const cl::ConstSharedPtr &cloud, const char *name, Stats &st, clk::time_point &dbg_t) {
        st.n++;
        st.last = clk::now();
        st.any = true;
        const bool dbg = due(dbg_t);

        geometry_msgs::msg::TransformStamped tf;
        try{
            tf = tf_buffer.lookupTransform(ground_frame, cloud->header.frame_id, tf2::TimePointZero);
        }
        catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[CLOUD %s] Lookup transform %s<-%s FAILED: %s", name, ground_frame.c_str(), cloud->header.frame_id.c_str(), e.what());
            return;
        }

        cl tc;
        tf2::doTransform(*cloud, tc, tf);

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

        const bool was_valid = lane.valid;
        lane.curve = fit_curve(pts, name, dbg);
        lane.stamp = rclcpp::Time(cloud->header.stamp);
        lane.valid = lane.curve.ok;

        if (!lane.valid) {
            st.fit_fail++;
            if (was_valid) st.flips++;
        }
    }

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

    Lane check(double y, bool has_white, double y_white, bool has_yellow, double y_yellow, bool dbg = false) const {
        double hi=0, lo=0;
        if (has_white && has_yellow) {
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

    // Publishes RViz MarkerArray for curves and detection bounding objects
    void publish_debug_markers(bool white_fresh, bool yellow_fresh, const std::vector<std::pair<geometry_msgs::msg::PointStamped, Lane>> &evaluated_detections) {
        visualization_msgs::msg::MarkerArray msg_array;
        int id = 0;

        // 1. White Lane Curve Marker (Line Strip)
        if (white_fresh && white.curve.ok) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = ground_frame;
            marker.header.stamp = this->now();
            marker.ns = "lane_curves";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.1; // Line width
            marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0;

            for (double x = white.curve.x_min; x <= white.curve.x_max; x += 0.5) {
                geometry_msgs::msg::Point p;
                p.x = x;
                p.y = white.curve.a + white.curve.b * x + white.curve.c * x * x;
                p.z = 0.0;
                marker.points.push_back(p);
            }
            msg_array.markers.push_back(marker);
        }

        // 2. Yellow Lane Curve Marker (Line Strip)
        if (yellow_fresh && yellow.curve.ok) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = ground_frame;
            marker.header.stamp = this->now();
            marker.ns = "lane_curves";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.1;
            marker.color.r = 1.0; marker.color.g = 0.9; marker.color.b = 0.0; marker.color.a = 1.0;

            for (double x = yellow.curve.x_min; x <= yellow.curve.x_max; x += 0.5) {
                geometry_msgs::msg::Point p;
                p.x = x;
                p.y = yellow.curve.a + yellow.curve.b * x + yellow.curve.c * x * x;
                p.z = 0.0;
                marker.points.push_back(p);
            }
            msg_array.markers.push_back(marker);
        }

        // 3. Object Positions with Lane-Status Color (Spheres)
        for (const auto &det_pair : evaluated_detections) {
            visualization_msgs::msg::Marker sphere;
            sphere.header.frame_id = ground_frame;
            sphere.header.stamp = this->now();
            sphere.ns = "evaluated_objects";
            sphere.id = id++;
            sphere.type = visualization_msgs::msg::Marker::SPHERE;
            sphere.action = visualization_msgs::msg::Marker::ADD;
            sphere.pose.position = det_pair.first.point;
            sphere.scale.x = 0.4; sphere.scale.y = 0.4; sphere.scale.z = 0.4;

            // Color coding: Green = IN, Red = OUT, Grey = UNKNOWN
            if (det_pair.second == Lane::IN) {
                sphere.color.r = 0.2; sphere.color.g = 0.8; sphere.color.b = 0.2; sphere.color.a = 0.9;
            } else if (det_pair.second == Lane::OUT) {
                sphere.color.r = 0.9; sphere.color.g = 0.2; sphere.color.b = 0.2; sphere.color.a = 0.9;
            } else {
                sphere.color.r = 0.5; sphere.color.g = 0.5; sphere.color.b = 0.5; sphere.color.a = 0.5;
            }
            msg_array.markers.push_back(sphere);
        }

        marker_pub->publish(msg_array);
    }

    void detection_callback(const det::ConstSharedPtr msg) {
        det_stats.n++;
        det_stats.last = clk::now();
        det_stats.any = true;
        const bool dbg = due(dbg_det_t);

        det output = *msg;
        const rclcpp::Time now(msg->header.stamp);
        const bool white_fresh = fresh(white, now);
        const bool yellow_fresh = fresh(yellow, now);
        const bool lanes_ok = white_fresh || yellow_fresh;

        if(lanes_ok == false) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Lane cloud missing");
        }

        bool have_tf = false;
        geometry_msgs::msg::TransformStamped g;
        if(lanes_ok && lookup(ground_frame, msg->header.frame_id, msg->header.stamp, g))
            have_tf = true;

        std::vector<std::pair<geometry_msgs::msg::PointStamped, Lane>> evaluated_detections;

        size_t n_in = 0, n_out = 0, n_unknown = 0, n_invalid = 0, n_notf = 0;
        size_t idx = 0;
        for (auto &item : output.detections) {
            const size_t i = idx++;
            const bool was_in_lane = item.is_in_lane;
            if(item.depth_valid == false) {
                item.is_in_lane = false;
                n_invalid++;
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

                r = check(target_point.point.y, has_white, y_white, has_yellow, y_yellow, dbg);
                evaluated_detections.push_back({target_point, r});
            } else {
                n_notf++;
            }
            item.is_in_lane = (r == Lane::IN) || (r == Lane::UNKNOWN && in_lane_unknown);
            if (r == Lane::IN) n_in++; else if (r == Lane::OUT) n_out++; else n_unknown++;
        }
        publisher->publish(output);
        publish_debug_markers(white_fresh, yellow_fresh, evaluated_detections);
        published++;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InLaneFilter>());
    rclcpp::shutdown();

    return 0;
}