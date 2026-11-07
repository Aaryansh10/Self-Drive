#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_interfaces/msg/detection_array.hpp>

#include "vision_cpp/gpu_pipeline.hpp"
#include "vision_cpp/ipm.hpp"
#include "vision_cpp/model_helpers.hpp"
#include "vision_cpp/thinning.hpp"
#include "vision_cpp/trt_engine.hpp"

namespace vision_cpp {

class VisionNode : public rclcpp::Node {
 public:
  VisionNode();

 private:
  using Clock = std::chrono::steady_clock;

  void img_cb(sensor_msgs::msg::Image::ConstSharedPtr msg);
  void imu_cb(sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void cam_info_cb(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

  void init_cloud(sensor_msgs::msg::PointCloud2& pc) const;
  void fill_cloud(sensor_msgs::msg::PointCloud2& pc, const std_msgs::msg::Header& h, size_t n);
  void report_profile();

  // parameters
  std::string ground_frame_;
  int in_w_, in_h_, step_;
  double cam_pitch_, cam_height_, max_range_;
  float nms_iou_;
  int max_dets_;
  std::vector<std::string> class_names_;
  bool profile_;
  int profile_period_;

  // GPU side
  std::unique_ptr<TrtEngine> engine_;
  std::unique_ptr<GpuPipeline> pipe_;

  // ROS
  rclcpp::CallbackGroup::SharedPtr cb_img_, cb_aux_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_white_, pub_yellow_, pub_stop_;
  rclcpp::Publisher<vision_interfaces::msg::DetectionArray>::SharedPtr pub_det_;

  // state shared between callback groups
  std::atomic<bool> cam_ready_{false};
  std::atomic<double> pitch_bot_{0.0};
  Ipm ipm_;

  // persistent messages and scratch (capacity reserved once -> no allocation per frame)
  sensor_msgs::msg::PointCloud2 pc_white_, pc_yellow_, pc_stop_;
  vision_interfaces::msg::DetectionArray det_msg_;
  DetScratch det_scratch_;
  SkelScratch skel_scratch_w_, skel_scratch_y_;
  std::vector<Detection> dets_;
  std::vector<int> vw_, uw_, vy_, uy_, vs_, us_;
  std::vector<float> pts_;

  // profiling (wall clock, no extra GPU syncs)
  double acc_copy_ = 0, acc_gpu_ = 0, acc_wait_ = 0, acc_cpu_ = 0, acc_pub_ = 0, acc_total_ = 0, acc_lat_ = 0;
  double max_total_ = 0;
  int prof_frames_ = 0;
  Clock::time_point last_report_ = Clock::now();
};

}  // namespace vision_cpp
