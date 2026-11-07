// bt executor: subscribes to the detections, builds the tree from XML, ticks it.
// path_bridge arbitrates lane/intersection paths and owns the Nav2 FollowPath client.
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_ros2/ros_node_params.hpp"
#include "fusion_interfaces/msg/tracked_obstacle_array.hpp"
#include "bt/nodes.hpp"
#include "bt/shared_state.hpp"
#include "bt/intersection_input.hpp"
#include "rclcpp/rclcpp.hpp"

using bt::Action;

static Action parse_action(const std::string &s, const rclcpp::Logger &log, const std::string &cls)
{
	if (s == "stop")
		return Action::STOP;
	if (s == "change")
		return Action::CHANGE;
	if (s == "intersection")
		return Action::INTERSECTION;
	if (s != "none")
		RCLCPP_WARN(log, "class '%s': unknown action '%s' (use stop | change | intersection | none)", cls.c_str(), s.c_str());
	return Action::NONE;
}

int main(int argc, char **argv)
{
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
	const auto intersection_action = node->declare_parameter<std::string>("intersection_action", "intersection");
	const double intersection_retry = node->declare_parameter<double>("intersection_min_retry", 2.0);
	const double intersection_server_wait = node->declare_parameter<double>("intersection_server_wait_timeout", 5.0);
	if (!std::isfinite(intersection_server_wait) || intersection_server_wait < 0.0) {
		RCLCPP_FATAL(log, "intersection_server_wait_timeout must be finite and nonnegative");
		return 1;
	}
	const double road_width = node->declare_parameter<double>("road_width", 3.0);
	const auto turn_names = node->declare_parameter<std::vector<std::string>>("intersection_turns", std::vector<std::string>{});

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
	cfg.max_lateral_offset = node->declare_parameter<double>("max_lateral_offset", 1.2);
	cfg.confirm_intersection = node->declare_parameter<int>("confirm_intersection", cfg.confirm_intersection);
	cfg.intersection_hold    = node->declare_parameter<double>("intersection_hold", cfg.intersection_hold);
	cfg.stop_sign_ttl        = node->declare_parameter<double>("stop_sign_ttl", cfg.stop_sign_ttl);

	// class table: class_name + classes.<name>.action / classes.<name>.trigger_distance
	const auto names = node->declare_parameter<std::vector<std::string>>("class_name", std::vector<std::string>{});
	for (const auto &n : names)
	{
		bt::ClassRule r;
		r.action = parse_action(node->declare_parameter<std::string>("classes." + n + ".action", "none"), log, n);
		r.mark_stop_sign = node->declare_parameter<bool>("classes." + n + ".mark_stop_sign", false);
		r.trigger_distance = node->declare_parameter<double>("classes." + n + ".trigger_distance", 0.0);
		cfg.classes[n] = r;
		RCLCPP_INFO(log, "class %-12s -> %-6s inside %.2f m", n.c_str(), bt::to_string(r.action), r.trigger_distance);
	}
	if (cfg.classes.empty())
		RCLCPP_WARN(log, "class_name is empty: nothing will ever trigger");
	if (tree_file.empty())
	{
		RCLCPP_FATAL(log, "parameter tree_file is empty");
		return 1;
	}

	auto state = std::make_shared<bt::SharedState>(
		cfg, stale, 
		[node] { 
			return node->now().seconds(); 
		}
	);
	
	using DetArray = fusion_interfaces::msg::TrackedObstacleArray;
	using IntersectionAction = intersection_interfaces_cpp::action::Intersection;

	std::vector<uint8_t> turns;
	for (const auto &t : turn_names) 
	{
		if (t == "straight")   
			turns.push_back(IntersectionAction::Goal::TURN_STRAIGHT);
		else if (t == "left")  
			turns.push_back(IntersectionAction::Goal::TURN_LEFT);
		else if (t == "right") 
			turns.push_back(IntersectionAction::Goal::TURN_RIGHT);
		else 
		{ 
			RCLCPP_FATAL(log, "bad turn '%s' (straight|left|right)", t.c_str()); 
			return 1; 
		}
	}
	state->set_turns(turns);
	bt::IntersectionInput intersection_input(node, state);

	auto sub = node->create_subscription<DetArray>(
		det_topic, rclcpp::SensorDataQoS(), 
		[state](DetArray::ConstSharedPtr msg) {
			std::vector<bt::Detection> dets;
			dets.reserve(msg->obstacles.size());
			for (const auto& it : msg->obstacles)
			{
				bt::Detection d;
				d.cls = it.class_name;
				d.score = 1.0; 
				d.n_pts = static_cast<int>(it.hits);
				d.valid = true; 
				d.distance = it.range;
				d.id = static_cast<int>(it.id);
				
				// Use position_odom so spatial memory stays fixed in world frame!
				d.x = it.position_odom.x;  
				d.y = it.position_odom.y;  
				
				// Keep local y for lateral lane filtering
				d.local_y = it.position_base.y; 

				dets.push_back(d);
			}
			state->on_detections(dets); 
		}
	);

	auto lane_gate = node->create_publisher<std_msgs::msg::Bool>(
		enable_topic, rclcpp::QoS(1).reliable().transient_local());
	lane_gate->publish(std_msgs::msg::Bool{});
	BT::BehaviorTreeFactory factory;
	factory.registerNodeType<bt::DecideAction>("DecideAction", state);
	factory.registerNodeType<bt::StopRobot>("StopRobot");
	factory.registerNodeType<bt::BeginIntersection>("BeginIntersection", state);
	factory.registerNodeType<bt::MarkIntersectionStopped>("MarkIntersectionStopped", state);
	factory.registerNodeType<bt::RetryOnNextTick>("RetryOnNextTick");
	factory.registerNodeType<bt::DisableLaneFollowing>("DisableLaneFollowing", lane_gate);
	factory.registerNodeType<bt::LaneFollowing>("LaneFollowing", lane_gate);
	factory.registerNodeType<bt::LaneChange>("LaneChange", BT::RosNodeParams(node, lane_change_action), state, min_retry);
	BT::RosNodeParams intersection_params(node, intersection_action);
	intersection_params.wait_for_server_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::duration<double>(intersection_server_wait));
	factory.registerNodeType<bt::IntersectionTurn>("Intersection", intersection_params, state, road_width, intersection_retry);

	BT::Tree tree = factory.createTreeFromFile(tree_file);
	RCLCPP_INFO(log, "bt running: tree = %s, %.0f Hz", tree_file.c_str(), tick_rate);

	rclcpp::Rate rate(tick_rate);
	while (rclcpp::ok())
	{
		rclcpp::spin_some(node);
		tree.tickOnce();
		rate.sleep();
	}
	tree.haltTree();
	rclcpp::shutdown();
	return 0;
}
