// BT nodes for bt. To add a new behaviour (intersection, parking) copy LaneChange below:
// one small class per action server, then register it in bt_executor.cpp and use it in trees/main.xml.
#pragma once
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_ros2/bt_action_node.hpp"
#include "bt/shared_state.hpp"
#include "lane_change_interfaces/action/lane_change.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace bt {

// Reads the latest decision and writes it to the blackboard: decision = none | change | stop.
class DecideAction : public BT::SyncActionNode {
 public:
  DecideAction(const std::string& name, const BT::NodeConfig& config, std::shared_ptr<SharedState> state)
      : BT::SyncActionNode(name, config), state_(std::move(state)) {}

  static BT::PortsList providedPorts() {
    return {BT::OutputPort<std::string>("decision"), BT::OutputPort<int>("obstacle_id")};
  }

  BT::NodeStatus tick() override {
    const Decision d = state_->current();
    setOutput("decision", std::string(to_string(d.action)));
    setOutput("obstacle_id", static_cast<int>(d.obstacle_id));
    if (d.action != last_) {
      RCLCPP_INFO(rclcpp::get_logger("bt"), "decision: %s -> %s (%s at %.2f m)", to_string(last_),
                  to_string(d.action), d.cls.c_str(), d.distance);
      last_ = d.action;
    }
    return BT::NodeStatus::SUCCESS;
  }

 private:
  std::shared_ptr<SharedState> state_;
  Action last_ = Action::NONE;
};

// Placeholder until a stop action server exists: reports it and FAILS, so the tree falls through to lane following.
class StopRobot : public BT::SyncActionNode {
 public:
  StopRobot(const std::string& name, const BT::NodeConfig& config) : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus tick() override {
    const auto t = std::chrono::steady_clock::now();
    if (t - last_log_ > std::chrono::seconds(2)) {
      RCLCPP_WARN(rclcpp::get_logger("bt"), "STOP requested but no stop action server found (placeholder)");
      last_log_ = t;
    }
    return BT::NodeStatus::FAILURE;
  }

 private:
  std::chrono::steady_clock::time_point last_log_{};
};

// Default behaviour: tells path_bridge lane following is enabled while this node runs, disabled when it is halted.
// (Needs the enable gate in path_bridge; without it the messages are simply ignored.)
class LaneFollowing : public BT::StatefulActionNode {
 public:
  LaneFollowing(const std::string& name, const BT::NodeConfig& config, rclcpp::Node::SharedPtr node,
                const std::string& topic)
      : BT::StatefulActionNode(name, config) {
    pub_ = node->create_publisher<std_msgs::msg::Bool>(topic, rclcpp::QoS(1).reliable().transient_local());
  }
  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus onStart() override {
    publish(true);
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override { return BT::NodeStatus::RUNNING; }
  void onHalted() override { publish(false); }

 private:
  void publish(bool enable) {
    std_msgs::msg::Bool m;
    m.data = enable;
    pub_->publish(m);
  }
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
};

// Sends ONE LaneChange goal. On success the obstacle is marked handled, so the same obstacle can't trigger again.
class LaneChange : public BT::RosActionNode<lane_change_interfaces::action::LaneChange> {
 public:
  using LaneChangeAction = lane_change_interfaces::action::LaneChange;

  LaneChange(const std::string& name, const BT::NodeConfig& conf, const BT::RosNodeParams& params,
             std::shared_ptr<SharedState> state, double min_retry_s)
      : RosActionNode<LaneChangeAction>(name, conf, params), state_(std::move(state)), min_retry_(min_retry_s) {}

  static BT::PortsList providedPorts() {
    return providedBasicPorts({BT::InputPort<int>("obstacle_id", -1, "obstacle that triggered this lane change")});
  }

  // Called once when the node starts. Sends a goal only for an active CHANGE decision, and not while a
  // recent FAILURE is still in its retry lockout. (A success never blocks the next, different obstacle.)
  bool setGoal(Goal& goal) override {
    if (state_->now() - last_fail_ < min_retry_) return false;
    const Decision d = state_->current();
    if (d.action != Action::CHANGE) return false;  // the obstacle went away before we could start
    sent_ = d;                                     // remember exactly which obstacle this goal is for
    goal.obstacle_distance = std::isfinite(d.distance) ? d.distance : 0.0;  // the server ignores these fields
    goal.completion_distance = 0.0;
    goal.target_speed = 0.0;
    goal.target_lane_free = true;
    return true;
  }

  BT::NodeStatus onResultReceived(const WrappedResult& wr) override {
    if (wr.result && wr.result->code == LaneChangeAction::Result::SUCCESS) {
      state_->mark_handled(sent_);  // handled only after the server confirmed success
      RCLCPP_INFO(logger(), "lane change done for obstacle %ld (%s): %s", sent_.obstacle_id, sent_.cls.c_str(),
                  wr.result->message.c_str());
      return BT::NodeStatus::SUCCESS;
    }
    last_fail_ = state_->now();
    RCLCPP_WARN(logger(), "lane change failed for obstacle %ld", sent_.obstacle_id);
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override {
    if (error == BT::INVALID_GOAL) return BT::NodeStatus::FAILURE;  // lockout / no active obstacle: expected, quiet
    last_fail_ = state_->now();
    RCLCPP_WARN(logger(), "LaneChange: %s (retry in %.1f s)", BT::toStr(error), min_retry_);
    return BT::NodeStatus::FAILURE;
  }

  // halt(): the wrapper cancels the goal for us; nothing was marked handled, so the obstacle can still trigger.
  void onHalt() override { RCLCPP_DEBUG(logger(), "LaneChange halted"); }

 private:
  std::shared_ptr<SharedState> state_;
  double min_retry_;
  double last_fail_ = -1e9;
  Decision sent_;
};

}  // namespace bt