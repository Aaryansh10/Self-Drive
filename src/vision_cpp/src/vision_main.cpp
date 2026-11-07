#include <rclcpp/rclcpp.hpp>

#include "vision_cpp/vision_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<vision_cpp::VisionNode>();
    // Two threads: one for the image callback group, one for IMU / camera info.
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
    exec.add_node(node);
    exec.spin();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "%s", e.what());
    code = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return code;
}
