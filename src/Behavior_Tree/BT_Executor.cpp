#include <chrono>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/blackboard.h"

#include "custom_msgs/msg/world_model.hpp"

using namespace std::chrono_literals;

class BTExecutor : public rclcpp::Node
{
public:

    BTExecutor()
        : Node("bt_executor") {
        world_model_sub_ = this->create_subscription<custom_msgs::msg::WorldModel>("/world_model", 10, std::bind(
                    &BTExecutor::worldModelCallback, this, std::placeholders::_1));

        factory_ = std::make_unique<BT::BehaviorTreeFactory>();

        blackboard_ = BT::Blackboard::create();

        tree_ = factory_->createTreeFromFile( "behavior_tree.xml", blackboard_);

        timer_ = this->create_wall_timer(100ms, std::bind( &BTExecutor::tickTree, this));
    }

private:

    void worldModelCallback(
        const custom_msgs::msg::WorldModel::SharedPtr msg)
    {
        blackboard_->set("world_model", msg);
    }

    void tickTree() {
        try {
            BT::NodeStatus status = tree_.tickOnce();

            switch (status)  {
                case BT::NodeStatus::RUNNING:
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "BT status: RUNNING");
                    break;

                case BT::NodeStatus::SUCCESS:
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "BT status: SUCCESS");
                    break;

                case BT::NodeStatus::FAILURE:
                    RCLCPP_WARN(
                        this->get_logger(),
                        "BT status: FAILURE");
                    break;

                case BT::NodeStatus::IDLE:
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "BT status: IDLE");
                    break;
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "BT execution error: %s",
                e.what());
        }
    }

    std::unique_ptr<BT::BehaviorTreeFactory> factory_;
    BT::Blackboard::Ptr blackboard_;
    BT::Tree tree_;
    rclcpp::Subscription<custom_msgs::msg::WorldModel>::SharedPtr world_model_sub_;

    rclcpp::TimerBase::SharedPtr timer_;
};

int main(
    int argc,
    char * argv[])
{
    rclcpp::init(argc, argv);

    auto node =
        std::make_shared<BTExecutor>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}