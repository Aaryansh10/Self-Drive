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
            const auto white_topic  = declare_parameter<std::string>("white_topic", "/pc_whitelane");
            const auto yellow_topic = declare_parameter<std::string>("yellow_topic", "/pc_yellowlane");
            const auto stop_topic   = declare_parameter<std::string>("stop_topic", "/pc_stopline");
            const auto map_topic    = declare_parameter<std::string>("map_topic", "/lane_map/white");
            odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
            base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
            publish_every_ = declare_parameter<int>("publish_every", 3);
            tf_timeout_ms_ = declare_parameter<int>("tf_timeout_ms", 50);
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
};
