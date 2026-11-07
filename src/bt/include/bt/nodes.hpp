// BT nodes for bt. To add a new behaviour (intersection, parking) copy LaneChange below:
// one small class per action server, then register it in bt_executor.cpp and use it in trees/main.xml.

// each new behaviour
// A small RosActionNode class in nodes.hpp (copy of LaneChange).
// Register it in bt_executor.cpp.
// Add a branch in main.xml.
// Add its parameters to bt.yaml.
// Add its interface package to CMakeLists.txt and package.xml.

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
#include "intersection_interfaces_cpp/action/intersection.hpp"

namespace bt
{

	// reads the latest decision and writes it on the blackboard  , decision = none  or change or stop.
	class DecideAction : public BT::SyncActionNode
	{
	public:
		DecideAction(const std::string &name, const BT::NodeConfig &config, std::shared_ptr<SharedState> state)
			: BT::SyncActionNode(name, config), state_(std::move(state)) {}

		static BT::PortsList providedPorts()
		{
			return {
				BT::OutputPort<std::string>("decision"), 
				BT::OutputPort<int>("obstacle_id"),
				BT::OutputPort<bool>("needs_stop"),
				BT::OutputPort<double>("distance")
			};
		}

		BT::NodeStatus tick() override
		{
			const Decision d = state_->current();
			setOutput("decision", std::string(to_string(d.action)));
			setOutput("obstacle_id", static_cast<int>(d.obstacle_id));
			setOutput("needs_stop", static_cast<bool>(d.needs_stop));
			setOutput("distance", d.distance);

			if (d.action != last_)
			{
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

    // Retry failed asynchronous work on a later tree tick, without falling
    // through to lane following or spinning in an unbounded retry loop.
    class RetryOnNextTick : public BT::DecoratorNode {
    public:
        RetryOnNextTick(const std::string& name, const BT::NodeConfig& config)
            : BT::DecoratorNode(name, config) {}
        static BT::PortsList providedPorts() { return {}; }
        BT::NodeStatus tick() override {
            setStatus(BT::NodeStatus::RUNNING);
            const auto result = child_node_->executeTick();
            if (result == BT::NodeStatus::FAILURE) {
                resetChild();
                return BT::NodeStatus::RUNNING;
            }
            if (result == BT::NodeStatus::SUCCESS) resetChild();
            return result;
        }
    };

    class BeginIntersection : public BT::SyncActionNode {
    public:
        BeginIntersection(const std::string& name, const BT::NodeConfig& config,
                          std::shared_ptr<SharedState> state)
            : BT::SyncActionNode(name, config), state_(std::move(state)) {}
        static BT::PortsList providedPorts() { return {BT::OutputPort<bool>("needs_stop")}; }
        BT::NodeStatus tick() override {
            const auto d = state_->begin_intersection();
            setOutput("needs_stop", d.needs_stop);
            return d.action == Action::INTERSECTION ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        }
    private:
        std::shared_ptr<SharedState> state_;
    };

    class MarkIntersectionStopped : public BT::SyncActionNode {
    public:
        MarkIntersectionStopped(const std::string& name, const BT::NodeConfig& config,
                                std::shared_ptr<SharedState> state)
            : BT::SyncActionNode(name, config), state_(std::move(state)) {}
        static BT::PortsList providedPorts() { return {BT::OutputPort<bool>("needs_stop")}; }
        BT::NodeStatus tick() override {
            state_->mark_intersection_stopped();
            setOutput("needs_stop", false);
            return BT::NodeStatus::SUCCESS;
        }
    private:
        std::shared_ptr<SharedState> state_;
    };

    class DisableLaneFollowing : public BT::SyncActionNode {
    public:
        DisableLaneFollowing(const std::string& name, const BT::NodeConfig& config,
                             rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher)
            : BT::SyncActionNode(name, config), pub_(std::move(publisher)) {}
        static BT::PortsList providedPorts() { return {}; }
        BT::NodeStatus tick() override {
            std_msgs::msg::Bool msg;
            msg.data = false;
            pub_->publish(msg);
            return BT::NodeStatus::SUCCESS;
        }
    private:
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
    };

	// Placeholder: returning FAILURE keeps the lane gate closed. Replace with
    // the stop action client; only SUCCESS permits the intersection goal.
	class StopRobot : public BT::SyncActionNode
	{
	public:
		StopRobot(const std::string &name, const BT::NodeConfig &config) : BT::SyncActionNode(name, config) {}
		static BT::PortsList providedPorts() { return {BT::InputPort<double>("distance", 0.0, "Distance to stop target; intersection stops use the stop line")}; }

		BT::NodeStatus tick() override
		{
			const auto t = std::chrono::steady_clock::now();
			if (t - last_log_ > std::chrono::seconds(2))
			{
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
	class LaneFollowing : public BT::StatefulActionNode
	{
	public:
        LaneFollowing(const std::string &name, const BT::NodeConfig &config,
                      rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher)
			: BT::StatefulActionNode(name, config), pub_(std::move(publisher)) {}
		static BT::PortsList providedPorts() { return {}; }

		BT::NodeStatus onStart() override
		{
			publish(true);
			return BT::NodeStatus::RUNNING;
		}
		BT::NodeStatus onRunning() override { return BT::NodeStatus::RUNNING; }
		void onHalted() override { publish(false); }

	private:
		void publish(bool enable)
		{
			std_msgs::msg::Bool m;
			m.data = enable;
			pub_->publish(m);
		}
		rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
	};

	// send lane change goal ... if success the obstacle is marked handled, so the same obstacle can't trigger again.
	class LaneChange : public BT::RosActionNode<lane_change_interfaces::action::LaneChange>
	{
	public:
		using LaneChangeAction = lane_change_interfaces::action::LaneChange;

		LaneChange(const std::string &name, const BT::NodeConfig &conf, const BT::RosNodeParams &params,
				   std::shared_ptr<SharedState> state, double min_retry_s)
			: RosActionNode<LaneChangeAction>(name, conf, params), state_(std::move(state)), min_retry_(min_retry_s) {}

		static BT::PortsList providedPorts()
		{
			return providedBasicPorts({BT::InputPort<int>("obstacle_id", -1, "obstacle that triggered this lane change")});
		}

		// Called once when the node starts. Sends a goal only for an active CHANGE decision, and not while a
		// recent FAILURE is still in its retry lockout. (A success never blocks the next, different obstacle.)
		bool setGoal(Goal &goal) override
		{
			if (state_->now() - last_fail_ < min_retry_)
				return false;
			const Decision d = state_->current();
			if (d.action != Action::CHANGE)
				return false;													   // the obstacle went away before we could start
			sent_ = d;															   // remember exactly which obstacle this goal is for
			goal.obstacle_distance = std::isfinite(d.distance) ? d.distance : 0.0; // the server ignores these fields
			goal.completion_distance = 0.0;
			goal.target_speed = 0.0;
			goal.target_lane_free = true;
			return true;
		}

		BT::NodeStatus onResultReceived(const WrappedResult &wr) override
		{
			if (wr.result && wr.result->code == LaneChangeAction::Result::SUCCESS)
			{
				state_->mark_handled(sent_); // handled only after the server confirmed success
				RCLCPP_INFO(logger(), "lane change done for obstacle %ld (%s): %s", sent_.obstacle_id, sent_.cls.c_str(),
							wr.result->message.c_str());
				return BT::NodeStatus::SUCCESS;
			}
			last_fail_ = state_->now();
			RCLCPP_WARN(logger(), "lane change failed for obstacle %ld", sent_.obstacle_id);
			return BT::NodeStatus::FAILURE;
		}

		BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override
		{
			if (error == BT::INVALID_GOAL)
				return BT::NodeStatus::FAILURE; // lockout / no active obstacle: expected, quiet
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

	class IntersectionTurn : public BT::RosActionNode<intersection_interfaces_cpp::action::Intersection>
	{
		public:
			using IntersectionAction = intersection_interfaces_cpp::action::Intersection;

			explicit IntersectionTurn(const std::string& name, const BT::NodeConfig& config, const BT::RosNodeParams& params,
									std::shared_ptr<SharedState> state, double road_width, double min_retry)
				: RosActionNode<IntersectionAction>(name, config, params), state_(std::move(state)), road_width_(road_width), min_retry_(min_retry) {}

			static BT::PortsList providedPorts() {
				return providedBasicPorts({});
			}

			bool setGoal(Goal& goal) override
			{
				if (state_->now() - last_fail_ < min_retry_) return false;
				const Decision d = state_->active_intersection();
				if (d.action != Action::INTERSECTION || d.needs_stop)
					return false;
				uint8_t turn = 0;

				if (!state_->peek_turn(turn))
				{
					last_fail_ = state_->now();
					RCLCPP_ERROR(logger(), "Intersection reached but turn list is empty.");
					return false;
				}

				sent_ = d;
				goal.turn_direction = turn;
				goal.road_width = static_cast<float>(road_width_);
				return true;
			}

			BT::NodeStatus onResultReceived(const WrappedResult &wr) override
			{
				if (wr.code == rclcpp_action::ResultCode::SUCCEEDED && wr.result && wr.result->success) {
					state_->complete_intersection(sent_);
					return BT::NodeStatus::SUCCESS;
				}

				last_fail_ = state_->now();
				RCLCPP_WARN(logger(), "intersection failed: %s", wr.result ? wr.result->message.c_str() : "?");
				return BT::NodeStatus::FAILURE;
			}

			BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override {
				if (error != BT::INVALID_GOAL) {
                        last_fail_ = state_->now();
                        RCLCPP_WARN(logger(), "Intersection: %s", BT::toStr(error));
                    }
				return BT::NodeStatus::FAILURE;
			}
  
			void onHalt() override {}   // the wrapper cancels the goal; the server then clears the path
		
		private:
			std::shared_ptr<SharedState> state_;
			double road_width_;
                double min_retry_;
                double last_fail_ = -1e9;
			Decision sent_;
	};

} // namespace bt