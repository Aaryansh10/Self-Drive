#include <thread>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "intersection_interfaces_cpp/action/intersection.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

class IntersectionServer : public rclcpp::Node {
    public:
        using Action = intersection_interfaces_cpp::action::Intersection;
        using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;

        IntersectionServer() : Node("intersection_server") {
            using namespace std::placeholders;
            this->action_server_ = rclcpp_action::create_server<Action>(
                this, 
                "intersection", 
                std::bind(&IntersectionServer::handle_goal, this, _1, _2),
                std::bind(&IntersectionServer::handle_cancel, this, _1),
                std::bind(&IntersectionServer::handle_accepted, this, _1)
            );
        }
    
    private:
        rclcpp_action::Server<Action>::ConstSharedPtr action_server_;

        rclcpp_action::GoalResponse handle_goal(
            const rclcpp_action::GoalUUID& uuid, 
            std::shared_ptr<const Action::Goal> goal
        )
        {
            RCLCPP_INFO(
                this->get_logger(), 
                "Recieved turn request, turn code: %d, road width: %f", goal->turn_direction, goal->road_width
            );
            (void)uuid;
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }

        rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> goal_handle) {
            RCLCPP_INFO(this->get_logger(), "Recieved request to cancel goal");
            (void) goal_handle;
            return rclcpp_action::CancelResponse::ACCEPT;
        }

        void handle_accepted() {
            std::thread{std::bind(&IntersectionServer::execute, this, std::placeholders::_1)}.detach();

        }

        void execute(const std::shared_ptr<GoalHandle> goal_handle) {
            RCLCPP_INFO(this->get_logger(), "Executing turn");
            rclcpp::Rate loop_rate(10);

            const auto goal = goal_handle->get_goal();
            auto feedback = std::make_shared<Action::Feedback>();
            auto result = std::make_shared<Action::Result>();

            if (goal->turn_direction == Action::Goal::TURN_LEFT) 
                feedback->current_state = "Initiating left turn";
            else if (goal->turn_direction == Action::Goal::TURN_RIGHT) 
                feedback->current_state = "Initiating right turn";
            else if (goal->turn_direction == Action::Goal::TURN_STRAIGHT) 
                feedback->current_state = "Proceeding straight";
            else 
                feedback->current_state = "Unknown turn direction";
    
        }


};
