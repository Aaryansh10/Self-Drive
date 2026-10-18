#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_msgs/action/follow_path.hpp>
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
}