// bt executor: subscribes to the detections, builds the tree from XML, ticks it.
// It does NOT follow paths and never calls /follow_path: lane following stays in vision -> mux -> path_bridge -> Nav2.
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_ros2/ros_node_params.hpp"
#include "fusion_interfaces/msg/tracked_obstacle_array.hpp"
#include "bt/nodes.hpp"
#include "bt/shared_state.hpp"
#include "rclcpp/rclcpp.hpp"

using bt::Action;

static Action parse_action(const std::string& s, const rclcpp::Logger& log, const std::string& cls) {
  if (s == "stop") return Action::STOP;
  if (s == "change") return Action::CHANGE;
  if (s != "none") RCLCPP_WARN(log, "class '%s': unknown action '%s' (use stop | change | none)", cls.c_str(), s.c_str());
  return Action::NONE;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("bt");
  const auto log = node->get_logger();

  const auto tree_file = node->declare_parameter<std::string>("tree_file", "");
  const auto det_topic = node->declare_parameter<std::string>("detections_topic", "/tracked_obstacles");
  const auto enable_topic = node->declare_parameter<std::string>("lane_follow_enable_topic", "/lane_follow/enable");
  const auto lane_change_action = node->declare_parameter<std::string>("lane_change_action", "lane_change");
  const double tick_rate = node->declare_parameter<double>("tick_rate", 20.0);
  const double stale = node->declare_parameter<double>("stale_timeout", 1.0);
  const double min_retry = node->declare_parameter<double>("lane_change_min_retry", 2.0);

  bt::DecisionConfig cfg;
  cfg.min_score = node->declare_parameter<double>("min_score", cfg.min_score);
  cfg.min_points = node->declare_parameter<int>("min_points", cfg.min_points);
  cfg.min_distance = node->declare_parameter<double>("min_distance", cfg.min_distance);
  cfg.confirm_change = node->declare_parameter<int>("confirm_change", cfg.confirm_change);
  cfg.confirm_stop = node->declare_parameter<int>("confirm_stop", cfg.confirm_stop);
  cfg.hold_time = node->declare_parameter<double>("hold_time", cfg.hold_time);
  cfg.handled_ttl = node->declare_parameter<double>("handled_ttl", cfg.handled_ttl);
  cfg.pseudo_ttl = node->declare_parameter<double>("pseudo_ttl", cfg.pseudo_ttl);
  cfg.same_obstacle_tol = node->declare_parameter<double>("same_obstacle_tol", cfg.same_obstacle_tol);

  // class table: class_names + classes.<name>.action / classes.<name>.trigger_distance
  const auto names = node->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});
  for (const auto& n : names) {
    bt::ClassRule r;
    r.action = parse_action(node->declare_parameter<std::string>("classes." + n + ".action", "none"), log, n);
    r.trigger_distance = node->declare_parameter<double>("classes." + n + ".trigger_distance", 0.0);
    cfg.classes[n] = r;
    RCLCPP_INFO(log, "class %-12s -> %-6s inside %.2f m", n.c_str(), bt::to_string(r.action), r.trigger_distance);
  }
  if (cfg.classes.empty()) RCLCPP_WARN(log, "class_names is empty: nothing will ever trigger");
  if (tree_file.empty()) {
    RCLCPP_FATAL(log, "parameter tree_file is empty");
    return 1;
  }

  auto state = std::make_shared<bt::SharedState>(cfg, stale, [node] { return node->now().seconds(); });

  // detections -> shared state. TODO(obstacle id): when DetectionDistance gets an id field, set d.id = it.<field>.
  /*using DetArray = fusion_interfaces::msg::TrackedObstacleArray;
  auto sub = node->create_subscription<DetArray>(det_topic, rclcpp::SensorDataQoS(), [state](DetArray::ConstSharedPtr msg) {
    std::vector<bt::Detection> dets;
    dets.reserve(msg->items.size());
    for (const auto& it : msg->items) {
      bt::Detection d;
      d.cls = it.class_name;
      d.score = it.score;
      d.n_pts = static_cast<int>(it.n_pts);
      d.valid = it.valid;
      d.distance = it.distance;
      d.id = -1;
      dets.push_back(d);
    }
    state->on_detections(dets);
  });*/
  // detections -> shared state
  using DetArray = fusion_interfaces::msg::TrackedObstacleArray;
  auto sub = node->create_subscription<DetArray>(det_topic, rclcpp::SensorDataQoS(), [state](DetArray::ConstSharedPtr msg) {
    std::vector<bt::Detection> dets;
    dets.reserve(msg->obstacles.size());
    for (const auto& it : msg->obstacles) {
      bt::Detection d;
      d.cls = it.class_name;
      d.score = 1.0; // Default score since TrackedObstacle does not publish score
      d.n_pts = static_cast<int>(it.hits);
      d.valid = true; // Assume tracked obstacles received are valid
      d.distance = it.range;
      d.id = static_cast<int>(it.id);
      dets.push_back(d);
    }
    state->on_detections(dets);
  });

  BT::BehaviorTreeFactory factory;
  factory.registerNodeType<bt::DecideAction>("DecideAction", state);
  factory.registerNodeType<bt::StopRobot>("StopRobot");
  factory.registerNodeType<bt::LaneFollowing>("LaneFollowing", node, enable_topic);
  factory.registerNodeType<bt::LaneChange>("LaneChange", BT::RosNodeParams(node, lane_change_action), state,
                                                min_retry);

  BT::Tree tree = factory.createTreeFromFile(tree_file);
  RCLCPP_INFO(log, "bt running: tree = %s, %.0f Hz", tree_file.c_str(), tick_rate);

  rclcpp::Rate rate(tick_rate);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    tree.tickOnce();
    rate.sleep();
  }
  tree.haltTree();
  rclcpp::shutdown();
  return 0;
}