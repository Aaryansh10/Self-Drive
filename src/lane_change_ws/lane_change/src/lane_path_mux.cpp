// lane_path_mux: republishes ONE of the two existing centrelines as /path/selected_centerline.
//   lane 1 = RIGHT = /path/right_centerline
//   lane 2 = LEFT  = /path/left_centerline
// The selected lane is commanded by lane_change_server (std_msgs/Int32 on /lane_change/selected_lane).
// Paths are republished unchanged (same frame, stamp and poses). No TF, no odometry, no Nav2.
#include <array>
#include <memory>
#include <string>

#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

namespace {
constexpr int kRight = 1;
constexpr int kLeft = 2;
const char* lane_name(int lane) { return lane == kLeft ? "LEFT" : "RIGHT"; }
}  // namespace

class LanePathMux : public rclcpp::Node {
 public:
  LanePathMux() : Node("lane_path_mux") {
    const auto right_topic = declare_parameter<std::string>("right_centerline_topic", "/path/right_centerline");
    const auto left_topic = declare_parameter<std::string>("left_centerline_topic", "/path/left_centerline");
    const auto out_topic = declare_parameter<std::string>("selected_centerline_topic", "/path/selected_centerline");
    const auto lane_topic = declare_parameter<std::string>("selected_lane_topic", "/lane_change/selected_lane");
    selected_ = declare_parameter<std::string>("initial_lane", "right") == "left" ? kLeft : kRight;

    // best-effort subscriptions accept both reliable and best-effort publishers
    const auto sensor_qos = rclcpp::SensorDataQoS();
    right_sub_ = create_subscription<nav_msgs::msg::Path>(
        right_topic, sensor_qos, [this](nav_msgs::msg::Path::ConstSharedPtr m) { on_path(kRight, m); });
    left_sub_ = create_subscription<nav_msgs::msg::Path>(
        left_topic, sensor_qos, [this](nav_msgs::msg::Path::ConstSharedPtr m) { on_path(kLeft, m); });

    // latched: if the server started first, we still receive its current lane
    lane_sub_ = create_subscription<std_msgs::msg::Int32>(
        lane_topic, rclcpp::QoS(1).reliable().transient_local(),
        [this](std_msgs::msg::Int32::ConstSharedPtr m) { on_lane(m->data); });

    // reliable output: compatible with both reliable and best-effort subscribers
    pub_ = create_publisher<nav_msgs::msg::Path>(out_topic, rclcpp::QoS(1).reliable());

    RCLCPP_INFO(get_logger(), "lane_path_mux: %s + %s -> %s (selected = %s)", right_topic.c_str(), left_topic.c_str(),
                out_topic.c_str(), lane_name(selected_));
  }

 private:
  void on_path(int lane, nav_msgs::msg::Path::ConstSharedPtr m) {
    if (m->poses.empty()) return;  // ignore empty paths
    latest_[lane - 1] = m;
    if (lane == selected_) pub_->publish(*m);
  }

  void on_lane(int lane) {
    if (lane != kRight && lane != kLeft) {
      RCLCPP_WARN(get_logger(), "ignoring invalid lane %d (1 = right, 2 = left)", lane);
      return;
    }
    if (lane != selected_) RCLCPP_INFO(get_logger(), "selected lane: %s -> %s", lane_name(selected_), lane_name(lane));
    selected_ = lane;
    const auto& p = latest_[selected_ - 1];
    if (p) pub_->publish(*p);  // switch immediately
    else RCLCPP_WARN(get_logger(), "no %s centreline received yet; will publish when it arrives", lane_name(selected_));
  }

  int selected_ = kRight;
  std::array<nav_msgs::msg::Path::ConstSharedPtr, 2> latest_;  // [0] = right, [1] = left
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr right_sub_, left_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr lane_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LanePathMux>());
  rclcpp::shutdown();
  return 0;
}