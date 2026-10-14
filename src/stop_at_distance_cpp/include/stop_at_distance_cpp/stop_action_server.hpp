#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"

#include "stop_at_distance/stop_types.hpp"
#include "stop_interfaces/action/stop_at_distance.hpp"

namespace stop_at_distance
{

class StopActionServer : public rclcpp_lifecycle::LifecycleNode
{
public:
  using StopAction = stop_interfaces::action::StopAtDistance;
  using GoalHandle = rclcpp_action::ServerGoalHandle<StopAction>;
  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit StopActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  struct Params
  {
    double control_rate_hz{50.0};
    double feedback_rate_hz{10.0};
    double mission_timeout{30.0};

    double band_min{1.0};
    double band_max{1.5};

    double stop_speed_epsilon{0.05};
    double settle_threshold_s{0.5};
    double max_settle_time{10.0};
    double hold_duration{0.5};
    double abort_stop_timeout{3.0};

    double approach_gain{0.5};
    double max_speed_mps{1.0};

    std::string distance_topic{"/scan_distance"};
    std::string odom_topic{"/odom"};
    std::string cmd_vel_topic{"/cmd_vel_stop"};
  };

  struct ActiveMission
  {
    std::shared_ptr<GoalHandle> goal_handle;
    std::string goal_id;
    Phase current_phase{Phase::IDLE};

    double start_time{0.0};
    double phase_entry_time{0.0};
    double brake_start_time{0.0};
    double last_feedback_time{0.0};

    double target_band_min{0.0};
    double target_band_max{0.0};
    double timeout{30.0};

    double last_distance_m{std::numeric_limits<double>::quiet_NaN()};
    double time_to_stop{0.0};
    bool time_to_stop_recorded{false};

    Outcome pending_outcome{Outcome::INTERNAL_FAULT}; 
    std::string pending_reason;
  };


  void declare_parameters();
  bool load_parameters();

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const StopAction::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle);

  void control_tick();

  void transition_phase(Phase new_phase, const std::string & reason);
  void begin_stop(Outcome outcome, const std::string & reason);   
  void finish_mission(Outcome outcome, const std::string & reason);  
  void publish_feedback(double now_s, double dist, double vel);

  void release_mission(const std::string & reason);

  Params params;
  std::atomic<bool> active_{false};  

  rclcpp::CallbackGroup::SharedPtr action_group_;
  rclcpp::CallbackGroup::SharedPtr state_group_;
  rclcpp::CallbackGroup::SharedPtr timer_group_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp_action::Server<StopAction>::SharedPtr action_server_;

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr distance_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

  std::mutex state_mutex_;
  double current_distance_m_{std::numeric_limits<double>::quiet_NaN()};
  double current_speed_mps_{std::numeric_limits<double>::quiet_NaN()};
  bool have_distance_{false};   
  bool have_speed_{false};

  std::mutex mission_mutex_;
  bool goal_reserved_{false};   
  ActiveMission active_mission_;
  SuccessChecker success_checker_;
};

}  