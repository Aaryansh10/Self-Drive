#include <chrono>
#include <cstdint>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_msgs/action/follow_path.hpp>
#include <std_msgs/msg/bool.hpp>

// Owns the single Nav2 FollowPath client. Intersection paths have priority;
// lane updates cannot replace a maneuver until its publisher clears the path.
class PathBridge : public rclcpp::Node {
public:
    using FollowPath = nav2_msgs::action::FollowPath;
    using GoalHandle = rclcpp_action::ClientGoalHandle<FollowPath>;
    using Path = nav_msgs::msg::Path;

    PathBridge() : Node("path_bridge") {
        action_client_ = rclcpp_action::create_client<FollowPath>(this, "follow_path");
        const auto lane_topic = declare_parameter<std::string>("lane_path_topic", "/path/selected_centerline");
        const auto intersection_topic = declare_parameter<std::string>("intersection_path_topic", "/intersection_path");
        const auto enable_topic = declare_parameter<std::string>("lane_follow_enable_topic", "/lane_follow/enable");
        lane_max_age_ = declare_parameter<double>("lane_path_max_age", 1.0);
        update_period_ = declare_parameter<double>("lane_update_period", 1.0);
        lane_sub_ = create_subscription<Path>(lane_topic, 10, [this](Path::ConstSharedPtr msg) {
            lane_path_ = msg;
            lane_received_ = now().seconds();
        });
        intersection_sub_ = create_subscription<Path>(intersection_topic,
            rclcpp::QoS(1).reliable().transient_local(), [this](Path::ConstSharedPtr msg) {
                if (msg->poses.empty()) {
                    if (intersection_path_) cancel_current();
                    intersection_path_.reset();
                } else {
                    cancel_current();
                    intersection_path_ = msg;
                }
                dispatch();
            });
        enable_sub_ = create_subscription<std_msgs::msg::Bool>(enable_topic,
            rclcpp::QoS(1).reliable().transient_local(), [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
                lane_enabled_ = msg->data;
                // A lane-disable message may arrive after the intersection path.
                // Only cancel a lane goal here; the server owns maneuver cancellation.
                if (!lane_enabled_ && !intersection_path_) cancel_current();
                dispatch();
            });
        timer_ = create_wall_timer(std::chrono::milliseconds(50), [this] { dispatch(); });
    }

private:
    void cancel_current() {
        ++generation_; // Invalidate pending acceptance/result callbacks as well.
        if (active_goal_) action_client_->async_cancel_goal(active_goal_);
        active_goal_.reset();
        sent_path_.reset();
        last_send_ = -1e9;
    }

    void dispatch() {
        const double t = now().seconds();
        Path::ConstSharedPtr desired = intersection_path_;
        if (!desired && lane_enabled_ && t - lane_received_ <= lane_max_age_) desired = lane_path_;
        if (!desired || desired->poses.empty() || desired == sent_path_) return;
        // A maneuver starts immediately; its one-shot latched publication is
        // retried by this timer if Nav2 has not started yet.
        if (t - last_send_ < update_period_ || !action_client_->action_server_is_ready()) return;
        FollowPath::Goal goal;
        goal.path = *desired;
        goal.controller_id = "FollowPath";
        const auto generation = ++generation_;
        sent_path_ = desired;
        last_send_ = t;
        rclcpp_action::Client<FollowPath>::SendGoalOptions options;
        options.goal_response_callback = [this, generation](GoalHandle::SharedPtr handle) {
            if (generation != generation_) {
                if (handle) action_client_->async_cancel_goal(handle);
                return;
            }
            active_goal_ = handle;
            if (!handle) {
                sent_path_.reset();
                RCLCPP_WARN(get_logger(), "FollowPath goal rejected; will retry");
            }
        };
        options.result_callback = [this, generation](const GoalHandle::WrappedResult& result) {
            if (generation != generation_) return;
            active_goal_.reset();
            if (result.code == rclcpp_action::ResultCode::ABORTED) {
                RCLCPP_WARN(get_logger(), "FollowPath aborted");
                // Do not silently resume lane following during an intersection.
            }
        };
        action_client_->async_send_goal(goal, options);
    }

    rclcpp_action::Client<FollowPath>::SharedPtr action_client_;
    rclcpp::Subscription<Path>::SharedPtr lane_sub_, intersection_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Path::ConstSharedPtr lane_path_, intersection_path_, sent_path_;
    GoalHandle::SharedPtr active_goal_;
    bool lane_enabled_ = false;
    double lane_received_ = -1e9, last_send_ = -1e9;
    double lane_max_age_, update_period_;
    uint64_t generation_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathBridge>());
    rclcpp::shutdown();
    return 0;
}
