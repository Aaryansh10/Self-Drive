/*#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_msgs/action/follow_path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <chrono>

class PathBridge : public rclcpp::Node {
public:
using FollowPath = nav2_msgs::action::FollowPath;
using GoalHandleFollowPath = rclcpp_action::ClientGoalHandle<FollowPath>;

PathBridge() : Node("path_bridge"), min_update_period(std::chrono::milliseconds(1000))  {
            action_client_ = rclcpp_action::create_client<FollowPath>(this, "follow_path");

            last_update_time = this->now();
            left_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
"/path/selected_centerline", 10,
std::bind(&PathBridge::path_cb_, this, std::placeholders::_1)
            );

RCLCPP_INFO(this->get_logger(), "Path bridge node started.");
        }
private:
rclcpp_action::Client<FollowPath>::SharedPtr action_client_;
rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr left_path_sub_;
rclcpp::Duration min_update_period;
rclcpp::Time last_update_time;

void path_cb_(nav_msgs::msg::Path::ConstSharedPtr msg) {
auto current_time = this->now();
if (current_time - last_update_time < min_update_period) {
return; 
            }

if (!action_client_->wait_for_action_server(std::chrono::milliseconds(10))) {
return;
            }

auto goal_msg = FollowPath::Goal();
            goal_msg.path = *msg;
            goal_msg.controller_id = "FollowPath";

auto send_goal_options = rclcpp_action::Client<FollowPath>::SendGoalOptions();
            send_goal_options.result_callback = std::bind(&PathBridge::result_cb_, this, std::placeholders::_1);
            action_client_->async_send_goal(goal_msg, send_goal_options);
            last_update_time = current_time;
        }

void result_cb_(const GoalHandleFollowPath::WrappedResult & result) {
switch (result.code) {
case rclcpp_action::ResultCode::SUCCEEDED:
RCLCPP_INFO(this->get_logger(), "Robot reached the final point of the tracked path.");
break;
case rclcpp_action::ResultCode::CANCELED:
RCLCPP_DEBUG(this->get_logger(), "Previous path seamlessly preempted by new update.");
break;
case rclcpp_action::ResultCode::ABORTED:
RCLCPP_WARN(this->get_logger(), "Path execution aborted by MPPI controller.");
break;
default:
break;
            }
        }
};

int main(int argc, char ** argv) {
rclcpp::init(argc, argv);
rclcpp::spin(std::make_shared<PathBridge>());
rclcpp::shutdown();
return 0;
}*/

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_msgs/action/follow_path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <chrono>

class PathBridge : public rclcpp::Node {
public:
    using FollowPath = nav2_msgs::action::FollowPath;
    using GoalHandleFollowPath = rclcpp_action::ClientGoalHandle<FollowPath>;

    PathBridge() : Node("path_bridge"), min_update_period(std::chrono::milliseconds(1000)), lane_follow_enabled_(false) {
        action_client_ = rclcpp_action::create_client<FollowPath>(this, "follow_path");

        last_update_time = this->now();
        
        // Subscribe to path updates
        left_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/path/selected_centerline", 10,
            std::bind(&PathBridge::path_cb_, this, std::placeholders::_1)
        );

        // Subscribe to the BT LaneFollowing enable gate
        enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/lane_follow/enable", 10,
            std::bind(&PathBridge::enable_cb_, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Path bridge node started and synced with BT gate.");
    }

private:
    rclcpp_action::Client<FollowPath>::SharedPtr action_client_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr left_path_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
    
    rclcpp::Duration min_update_period;
    rclcpp::Time last_update_time;
    bool lane_follow_enabled_;
    GoalHandleFollowPath::SharedPtr active_goal_handle_;

    void enable_cb_(const std_msgs::msg::Bool::SharedPtr msg) {
        lane_follow_enabled_ = msg->data;
        if (!lane_follow_enabled_ && active_goal_handle_) {
            // Cancel current navigation goal if lane following is disabled by BT (e.g., during Lane Change)
            action_client_->async_cancel_goal(active_goal_handle_);
            active_goal_handle_.reset();
            RCLCPP_INFO(this->get_logger(), "Lane following disabled by BT; canceling active path execution.");
        }
    }

    void path_cb_(nav_msgs::msg::Path::ConstSharedPtr msg) {
        // If the behavior tree has disabled lane following, ignore incoming paths
        if (!lane_follow_enabled_) {
            return;
        }

        auto current_time = this->now();
        if (current_time - last_update_time < min_update_period) {
            return; 
        }

        if (!action_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "FollowPath action server not available.");
            return;
        }

        auto goal_msg = FollowPath::Goal();
        goal_msg.path = *msg;
        goal_msg.controller_id = "FollowPath";

        auto send_goal_options = rclcpp_action::Client<FollowPath>::SendGoalOptions();
        send_goal_options.result_callback = std::bind(&PathBridge::result_cb_, this, std::placeholders::_1);
        
        send_goal_options.goal_response_callback = 
            [this](typename rclcpp_action::Client<FollowPath>::GoalResponseCallback::argument_type goal_handle) {
                active_goal_handle_ = goal_handle;
            };

        action_client_->async_send_goal(goal_msg, send_goal_options);
        last_update_time = current_time;
    }

    void result_cb_(const GoalHandleFollowPath::WrappedResult & result) {
        active_goal_handle_.reset();
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Robot reached the final point of the tracked path.");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_DEBUG(this->get_logger(), "Previous path seamlessly preempted by new update or BT halt.");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_WARN(this->get_logger(), "Path execution aborted by MPPI controller.");
                break;
            default:
                break;
        }
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathBridge>());
    rclcpp::shutdown();
    return 0;
}