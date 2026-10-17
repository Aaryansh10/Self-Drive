/* / lane_path_mux: republishes ONE of the two existing centrelines as /path/selected_centerline.
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

    // best effort subscriptions accept both reliable and best effort publishers
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
*/

//lane_path_mux: republishes ONE of the two existing centrelines as /path/selected_centerline.
//   lane 1 = RIGHT = /path/right_centerline
//   lane 2 = LEFT  = /path/left_centerline
// The selected lane is commanded by lane_change_server (std_msgs/Int32 on /lane_change/selected_lane).
// The centreline shape is never changed. If output_frame is set (e.g. "odom"), the path is only
// re-expressed in that fixed frame using TF at the path's own stamp, so it stops moving with the robot.
// output_frame = "" -> pass-through (frame left as published by vision).
#include <array>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "tf2/time.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

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
    output_frame_ = declare_parameter<std::string>("output_frame", "");
    selected_ = declare_parameter<std::string>("initial_lane", "right") == "left" ? kLeft : kRight;

    if (!output_frame_.empty()) {
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    }

    const auto sensor_qos = rclcpp::SensorDataQoS();
    right_sub_ = create_subscription<nav_msgs::msg::Path>(
        right_topic, sensor_qos, [this](nav_msgs::msg::Path::ConstSharedPtr m) { on_path(kRight, m); });
    left_sub_ = create_subscription<nav_msgs::msg::Path>(
        left_topic, sensor_qos, [this](nav_msgs::msg::Path::ConstSharedPtr m) { on_path(kLeft, m); });
    lane_sub_ = create_subscription<std_msgs::msg::Int32>(
        lane_topic, rclcpp::QoS(1).reliable().transient_local(),
        [this](std_msgs::msg::Int32::ConstSharedPtr m) { on_lane(m->data); });
    pub_ = create_publisher<nav_msgs::msg::Path>(out_topic, rclcpp::QoS(1).reliable());

    RCLCPP_INFO(get_logger(), "lane_path_mux: %s + %s -> %s (selected = %s, output_frame = '%s')", right_topic.c_str(),
                left_topic.c_str(), out_topic.c_str(), lane_name(selected_), output_frame_.c_str());
  }

 private:
  void on_path(int lane, nav_msgs::msg::Path::ConstSharedPtr m) {
    if (m->poses.empty()) return;  // ignore empty paths
    latest_[lane - 1] = m;
    if (lane == selected_) publish(*m);
  }

  void on_lane(int lane) {
    if (lane != kRight && lane != kLeft) {
      RCLCPP_WARN(get_logger(), "ignoring invalid lane %d (1 = right, 2 = left)", lane);
      return;
    }
    if (lane != selected_) RCLCPP_INFO(get_logger(), "selected lane: %s -> %s", lane_name(selected_), lane_name(lane));
    selected_ = lane;
    const auto& p = latest_[selected_ - 1];
    if (p) publish(*p);
    else RCLCPP_WARN(get_logger(), "no %s centreline received yet; will publish when it arrives", lane_name(selected_));
  }

  // pass-through, or re-express in output_frame_ (same points, same order, only the frame changes)
  void publish(const nav_msgs::msg::Path& in) {
    if (output_frame_.empty() || in.header.frame_id == output_frame_) {
      pub_->publish(in);
      return;
    }
    geometry_msgs::msg::TransformStamped tf;
    try {  // exact: where the robot was when vision produced this path
      tf = tf_buffer_->lookupTransform(output_frame_, in.header.frame_id, in.header.stamp, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException&) {
      try {  // fallback: latest available transform
        tf = tf_buffer_->lookupTransform(output_frame_, in.header.frame_id, tf2::TimePointZero);
      } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF %s -> %s unavailable, path not published: %s",
                             in.header.frame_id.c_str(), output_frame_.c_str(), e.what());
        return;
      }
    }
    nav_msgs::msg::Path out;
    out.header = in.header;
    out.header.frame_id = output_frame_;
    out.poses.reserve(in.poses.size());
    for (const auto& p : in.poses) {
      geometry_msgs::msg::PoseStamped o;
      tf2::doTransform(p, o, tf);
      o.header.frame_id = output_frame_;
      o.header.stamp = in.header.stamp;
      out.poses.push_back(o);
    }
    pub_->publish(out);
  }

  std::string output_frame_;
  int selected_ = kRight;
  std::array<nav_msgs::msg::Path::ConstSharedPtr, 2> latest_;  // [0] = right, [1] = left
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
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
