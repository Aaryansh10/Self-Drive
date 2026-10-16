// lane_change_server: LaneChange action server that only flips the selected lane.
//   lane 1 = RIGHT, lane 2 = LEFT (published as std_msgs/Int32, latched, on /lane_change/selected_lane)
// It does not plan, transform, or drive anything. lane_path_mux does the path selection.
// Goal fields (obstacle_distance, completion_distance, target_speed, target_lane_free) are accepted and ignored.
#include <memory>
#include <string>

#include "lane_change_interfaces/action/lane_change.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/int32.hpp"

using Action = lane_change_interfaces::action::LaneChange;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;

namespace {
constexpr int kRight = 1;
constexpr int kLeft = 2;
const char* lane_name(int lane) { return lane == kLeft ? "LEFT" : "RIGHT"; }
}  // namespace

class LaneChangeServer : public rclcpp::Node {
 public:
  LaneChangeServer() : Node("lane_change_server") {
    const auto topic = declare_parameter<std::string>("selected_lane_topic", "/lane_change/selected_lane");
    lane_ = declare_parameter<std::string>("initial_lane", "right") == "left" ? kLeft : kRight;

    pub_ = create_publisher<std_msgs::msg::Int32>(topic, rclcpp::QoS(1).reliable().transient_local());
    publish_lane();

    server_ = rclcpp_action::create_server<Action>(
        this, "lane_change",
        [](const rclcpp_action::GoalUUID&, std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](std::shared_ptr<GoalHandle>) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](std::shared_ptr<GoalHandle> gh) {
          // instant operation: flip, publish, succeed
          lane_ = (lane_ == kRight) ? kLeft : kRight;
          publish_lane();
          RCLCPP_INFO(get_logger(), "lane change: selected lane = %s", lane_name(lane_));
          auto res = std::make_shared<Action::Result>();
          res->code = Action::Result::SUCCESS;
          res->message = std::string("selected lane = ") + lane_name(lane_);
          res->body_clear_distance = -1.0;  // not observed: no manoeuvre is performed
          res->in_window = false;
          gh->succeed(res);
        });
    RCLCPP_INFO(get_logger(), "lane_change server ready, selected lane = %s -> %s", lane_name(lane_), topic.c_str());
  }

 private:
  void publish_lane() {
    std_msgs::msg::Int32 m;
    m.data = lane_;
    pub_->publish(m);
  }

  int lane_ = kRight;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_;
  rclcpp_action::Server<Action>::SharedPtr server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneChangeServer>());
  rclcpp::shutdown();
  return 0;
}