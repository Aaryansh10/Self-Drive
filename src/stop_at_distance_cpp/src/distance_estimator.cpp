#include "stop_at_distance/distance_estimator.hpp"

#include <cmath>
#include <mutex>

namespace stop_at_distance {
    DistanceEstimator::DistanceEstimator(const rclcpp::NodeOptions & options) : Node("distance_estimator", options){
        odom_sub = create_subscription<nav_msgs::msg::Odometry> (
            "/odom", 10, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_speed = msg->twist.twist.linear.x;
            }
        );
        
        est_pub = create_publisher<stop_interfaces::msg::EstimatedDistance> (
            "/stop/estimated_distance", 10
        );

        distance_sub = create_subscription<std_msgs::msg::Float32> (
            "/scan_distance", 10, [this](const std_msgs::msg::Float32::SharedPtr msg) {
              double t_now = this->now().seconds();
              double current_distance = msg->data;
              double speed;
              {
                std::lock_guard<std::mutex>lock(state_mutex_);
                speed = current_speed;
              }

              if (last_time >= 0.0){
                double dt = t_now - last_time;
                if (dt > 0.001) {
                    double raw_closing_speed = (last_distance - current_distance) / dt;
                    filtered_closing_speed = (alpha * raw_closing_speed) + ((1.0 - alpha) * filtered_closing_speed);
                }
                }
              else {
                filtered_closing_speed = 0.0;
              }
              if (std::fabs(speed) < 0.02 && std::fabs(filtered_closing_speed) < 0.1) {
                filtered_closing_speed = 0.0;

            }

            last_distance = current_distance;
            last_time = t_now;
            stop_interfaces::msg::EstimatedDistance out_msg;
            out_msg.stamp = this->now();
            out_msg.distance = current_distance; 
            out_msg.closing_speed = filtered_closing_speed;
            out_msg.variance = 0.01;
            out_msg.valid = true;
            out_msg.source = "SIM_FUSED_TRACKER";
            out_msg.age = 0.0;

            est_pub->publish(out_msg);
    });

            RCLCPP_INFO(get_logger(), "Sim Distance Estimator (Pass-Through) Active.");
}
    }
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stop_at_distance::DistanceEstimator)