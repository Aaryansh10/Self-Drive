#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Vector3.h"
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
        x_window = declare_parameter<double>("x_window", 5.0);
        max_lane_width = declare_parameter<double>("max_lane_width", 5.0);
        min_lane_width = declare_parameter<double>("min_lane_width", 1.0);
        lane_width = declare_parameter<double>("lane_width", 3.8);
        lane_margin = declare_parameter<double>("lane_margin", 0.0);
        lane_max_age = declare_parameter<double>("lane_max_age", 0.5);
        in_lane_unknown = declare_parameter<bool>("in_lane_unknown", true);        

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

    struct LaneCloud {
        std::vector<P2> pts;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        bool valid = false;
    };

    enum class Lane {IN,OUT,UNKNOWN};
    std::string ground_frame;
    double x_window, max_lane_width, min_lane_width, lane_margin, lane_max_age, lane_width;
    bool in_lane_unknown;
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

        lane.pts = std::move(pts);
        lane.stamp = rclcpp::Time(cloud->header.stamp);
        lane.valid = !lane.pts.empty();
    }

    //Check whether data is fresh
    bool fresh(const LaneCloud &lane, const rclcpp::Time &now) const {
        return lane.valid && std::fabs((now - lane.stamp).seconds()) <= lane_max_age;
    }

    //searches for lane point closest to robot's centerline
    bool boundary(const LaneCloud &lane, double x, double &y_out) const {
        bool found = false;
        double best = std::numeric_limits<double>::infinity();
        for (const auto &p : lane.pts) {
            if (std::fabs(p.x - x) > x_window)
            continue;
            double d = std::fabs(p.y);
            if(d<best) {
                best = d;
                y_out = p.y;
                found = true;
            }
        }
        return found;
    }

    //Main function to check whether in lane
    Lane check(double x, double y, bool white_fresh, bool yellow_fresh) const {
        double y_white=0 , y_yellow=0;
        bool has_white = white_fresh && boundary(white, x, y_white);    
        bool has_yellow = yellow_fresh && boundary(yellow, x, y_yellow);
        double hi=0, lo=0;
        if (has_white && has_yellow) { 
            lo = std::min(y_white, y_yellow);
            hi = std::max(y_white, y_yellow);
        if(hi-lo > max_lane_width || hi-lo < min_lane_width) 
        return Lane::UNKNOWN;
        }
        else if(has_white || has_yellow) {
            double y_line = has_yellow ? y_yellow:y_white;
            bool left = y_line > 0;
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
        const::rclcpp::Time now(msg->header.stamp);
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
                r = check(target_point.point.x , target_point.point.y, white_fresh, yellow_fresh);
                //debug 
                double y_white = 0, y_yellow = 0;
                bool has_white = white_fresh && boundary(white, target_point.point.x, y_white);      //changed: respects freshness like check()
                bool has_yellow = yellow_fresh && boundary(yellow, target_point.point.x, y_yellow);  //changed: respects freshness like check()
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "obj x=%.2f y=%.2f z=%.2f | white=%d %.2f yellow=%d %.2f -> %d",   //changed: throttled to once per second
                target_point.point.x, target_point.point.y, target_point.point.z,
                    has_white, y_white, has_yellow, y_yellow, static_cast<int>(r));

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
