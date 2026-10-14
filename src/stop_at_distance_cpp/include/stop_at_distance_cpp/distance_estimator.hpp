#pragma once

#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "stop_interfaces/msg/estimated_distance.hpp"

namespace stop_at_distance {
    
    class DistanceEstimator : public rclcpp::Node {
        public:
        explicit DistanceEstimator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

        private:
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr distance_sub;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
        rclcpp::Publisher<stop_interfaces::msg::EstimatedDistance>::SharedPtr est_pub;

        std::mutex state_mutex_;
        double current_speed = 0.0;
        double last_time = -1.0;
        double last_distance = -1.0;
        double filtered_closing_speed = 0.0;
        double alpha = 0.2;


        };
    }