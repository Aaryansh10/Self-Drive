#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "lane_mapper.hpp"

using PC2 = sensor_msgs::msg::PointCloud2;
using SyncPolicy = message_filters::sync_policies::ExactTime<PC2, PC2, PC2>;

class MappingNode : public rclcpp::Node {
    public:
        MappingNode() : Node("mapping_node") {
            // Topics / frames
            const auto white_topic = declare_parameter<std::string>("white_topic", "/pc_whitelane");
            const auto yellow_topic = declare_parameter<std::string>("yellow_topic", "/pc_yellowlane");
            const auto stop_topic = declare_parameter<std::string>("stop_topic", "/pc_stopline");
            const auto map_topic = declare_parameter<std::string>("map_topic", "/lane_map/white");
            odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
            base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
            publish_every_ = declare_parameter<int>("publish_every", 3);
            tf_timeout_ms_ = declare_parameter<int>("tf_timeout_ms", 50);

            // Occupancy Grid Parameters
            lane_mapper::Params p;
            p.resolution = declare_parameter<double>("resolution", 0.1);
            p.width = declare_parameter<int>("width", 250);
            p.height = declare_parameter<int>("height", 250);
            p.roi_x_min = declare_parameter<double>("roi_x_min", 1.02);
            p.roi_x_max = declare_parameter<double>("roi_x_max", 7.72);
            p.roi_half_y_near = declare_parameter<double>("roi_half_y_near", 1.93);
            p.roi_half_y_far = declare_parameter<double>("roi_half_y_far", 11.02);
            p.sigma_base = declare_parameter<double>("sigma_base", 0.09);
            p.sigma_per_m = declare_parameter<double>("sigma_per_m", 0.0);
            p.sigma_per_m2 = declare_parameter<double>("sigma_per_m2", 0.002);
            p.range_ref = declare_parameter<double>("range_ref", 5.0);
            p.l_hit = declare_parameter<double>("l_hit", 0.9);
            p.l_miss = declare_parameter<double>("l_miss", -0.12);
            p.l_min = declare_parameter<double>("l_min", -2.0);
            p.l_max = declare_parameter<double>("l_max", 5.0);
            p.l_occupied = declare_parameter<double>("l_occ", 1.5);
            p.min_points_for_miss = declare_parameter<int>("min_points_for_miss", 30);  // combined count
            p.speckle_radius  = declare_parameter<int>("speckle_radius", 1);
            p.speckle_support = declare_parameter<int>("speckle_support", 3);
            roi_x_min_ = p.roi_x_min;
            roi_x_max_ = p.roi_x_max;
            grid_ = std::make_unique<lane_mapper::LaneMapper>(p);

            // TF
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
            tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);  // own thread
            
            // Sync clouds
            sub_w_.subscribe(this, white_topic, rmw_qos_profile_sensor_data);
            sub_y_.subscribe(this, yellow_topic, rmw_qos_profile_sensor_data);
            sub_s_.subscribe(this, stop_topic, rmw_qos_profile_sensor_data);
            sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), sub_w_, sub_y_, sub_s_);
            sync_->registerCallback(
                std::bind(&MappingNode::cb, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3)
            );
            map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic, rclcpp::QoS(1).reliable());
            RCLCPP_INFO(get_logger(), "Mapping node started.");
        }

    private:
        std::string odom_frame_, base_frame_;
        int publish_every_ = 3; 
        int tf_timeout_ms_ = 50;
        unsigned frame_ = 0;
        double roi_x_min_ = 0.0;
        double roi_x_max_ = 0.0;
        
        std::unique_ptr<lane_mapper::LaneMapper> grid_;
        std::vector<int8_t> buf_;
        lane_mapper::Window win_{};
        
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
        message_filters::Subscriber<PC2> sub_w_, sub_y_, sub_s_;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

        void cb(const PC2::ConstSharedPtr& white, const PC2::ConstSharedPtr& yellow, const PC2::ConstSharedPtr& stopline) {
            // Pose of the robot at "Image" time and not "now". Inference latency would smear lines otherwise
            geometry_msgs::msg::TransformStamped tf;
            try {
                tf = tf_buffer_->lookupTransform(
                    odom_frame_, base_frame_, white->header.stamp,
                    rclcpp::Duration(std::chrono::milliseconds(tf_timeout_ms_))
                );
            }
            catch (const tf2::TransformException& e){
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", e.what());
                return;
            }

            const auto& t = tf.transform.translation;
            const auto& q = tf.transform.rotation;
            const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            const lane_mapper::Pose2 pose{t.x, t.y, yaw};

            // Taking the white lane points inside the trusted range band
            std::vector<lane_mapper::Point2> pts;
            const size_t n = size_t(white->width) * white->height;
            pts.reserve(n);
            sensor_msgs::PointCloud2ConstIterator<float> ix(*white, "x"), iy(*white, "y");
            for (size_t i = 0; i < n; ++i, ++ix, ++iy) {
                const float x = *ix;
                const float y = *iy;
                if (!std::isfinite(x) || !std::isfinite(y))
                    continue;
                if (x > roi_x_max_ || x < roi_x_min_)
                continue;
                pts.push_back({x, y});
            }

            // Sensor reliabilty check
            const int health = int(n + size_t(yellow->width) * yellow->height + size_t(stopline->width) * stopline->height);
            
            grid_->insert(pts, pose, health);
            
            if (++frame_ % publish_every_ != 0)
                return;

            grid_->snapshot(buf_, win_);
            
            nav_msgs::msg::OccupancyGrid msg;
            msg.header.stamp = white->header.stamp;
            msg.header.frame_id = odom_frame_;
            msg.info.resolution = float(win_.resolution);
            msg.info.width = uint32_t(win_.width);
            msg.info.height = uint32_t(win_.height);
            msg.info.origin.position.x = win_.origin_x;
            msg.info.origin.position.y = win_.origin_y;
            msg.info.origin.orientation.w = 1.0;
            msg.data = buf_;
            map_pub_->publish(msg);
        }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MappingNode>());
    rclcpp::shutdown();
    return 0;
}