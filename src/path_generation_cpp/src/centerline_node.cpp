// Centerline node: subscribes to the lane point clouds published by the vision node and publishes the
// left/right lane centerline paths. Replaces the CenterlineGenerator + xy_to_path part of vision.py.
//
//   in : /pc_whitelane   sensor_msgs/PointCloud2  (ordered leading/skeleton points, ground frame)
//        /pc_yellowlane  sensor_msgs/PointCloud2
//   out: /path/left_centerline, /path/right_centerline   nav_msgs/Path
//
// The vision node publishes both clouds with the same header stamp, so they are paired by stamp.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "path_generation_cpp/centerline_generator.hpp"

using nav_msgs::msg::Path;
using sensor_msgs::msg::PointCloud2;

namespace {

std::vector<centerline::Point2> toPoints(const PointCloud2::ConstSharedPtr& msg) {
  std::vector<centerline::Point2> pts;
  if (!msg) return pts;
  const std::size_t n = static_cast<std::size_t>(msg->width) * msg->height;
  if (n == 0 || msg->data.empty()) return pts;  // the iterator dereferences data.front()

  pts.reserve(n);
  sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
  for (std::size_t i = 0; i < n; ++i, ++ix, ++iy) {
    const double x = *ix;
    const double y = *iy;
    if (std::isfinite(x) && std::isfinite(y)) pts.push_back({x, y});
  }
  return pts;
}

Path toPathMsg(const centerline::Path& pts, const std_msgs::msg::Header& header) {
  Path path;
  path.header = header;
  path.poses.reserve(pts.size());
  for (const auto& p : pts) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header;
    pose.pose.position.x = p.x;
    pose.pose.position.y = p.y;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(std::move(pose));
  }
  return path;
}

}  // namespace

class CenterlineNode : public rclcpp::Node {
 public:
  CenterlineNode() : rclcpp::Node("centerline") {
    const auto white_topic = declare_parameter<std::string>("pc_whitelane_topic", "/pc_whitelane");
    const auto yellow_topic = declare_parameter<std::string>("pc_yellowlane_topic", "/pc_yellowlane");
    const auto left_topic = declare_parameter<std::string>("left_centerline_topic", "/path/left_centerline");
    const auto right_topic = declare_parameter<std::string>("right_centerline_topic", "/path/right_centerline");
    timing_log_period_ = static_cast<int>(declare_parameter<int64_t>("timing_log_period", 0));

    gen_ = std::make_unique<centerline::CenterlineGenerator>(declareGeneratorParams());

    white_sub_ = create_subscription<PointCloud2>(
        white_topic, rclcpp::SensorDataQoS(),
        [this](PointCloud2::ConstSharedPtr msg) { onWhite(std::move(msg)); });
    yellow_sub_ = create_subscription<PointCloud2>(
        yellow_topic, rclcpp::SensorDataQoS(),
        [this](PointCloud2::ConstSharedPtr msg) { onYellow(std::move(msg)); });

    rclcpp::QoS path_qos(rclcpp::KeepLast(1));
    path_qos.reliable();
    left_pub_ = create_publisher<Path>(left_topic, path_qos);
    right_pub_ = create_publisher<Path>(right_topic, path_qos);

    RCLCPP_INFO(get_logger(), "Centerline node started (white: %s, yellow: %s).", white_topic.c_str(),
                yellow_topic.c_str());
  }

 private:
  centerline::Params declareGeneratorParams() {
    centerline::Params p;  // defaults == CenterlineGenerator.__init__ defaults
    auto d = [this](const char* name, double def) { return declare_parameter<double>(name, def); };
    auto i = [this](const char* name, int def) {
      return static_cast<int>(declare_parameter<int64_t>(name, def));
    };

    p.max_gap = d("max_gap", p.max_gap);
    p.bin_width = d("bin_width", p.bin_width);
    p.max_bad_frame_streak = i("max_bad_frame_streak", p.max_bad_frame_streak);
    p.max_range = d("max_range", p.max_range);
    p.nominal_road_width = d("nominal_road_width", p.nominal_road_width);
    p.min_road_width = d("min_road_width", p.min_road_width);
    p.max_road_width = d("max_road_width", p.max_road_width);
    p.min_bin_points = i("min_bin_points", p.min_bin_points);
    p.max_width_update = d("max_width_update", p.max_width_update);
    p.min_offset_fraction = d("min_offset_fraction", p.min_offset_fraction);
    p.yellow_gate_fraction = d("yellow_gate_fraction", p.yellow_gate_fraction);
    p.path_spacing = d("path_spacing", p.path_spacing);
    p.path_start = d("path_start", p.path_start);
    p.final_smooth_window = i("final_smooth_window", p.final_smooth_window);
    p.near_field_range = d("near_field_range", p.near_field_range);
    p.min_near_pts = i("min_near_pts", p.min_near_pts);
    p.width_stale_decay_frames = i("width_stale_decay_frames", p.width_stale_decay_frames);
    p.width_stale_decay_alpha = d("width_stale_decay_alpha", p.width_stale_decay_alpha);
    p.sg_polyorder = i("sg_polyorder", p.sg_polyorder);
    p.track_ema_alpha = d("track_ema_alpha", p.track_ema_alpha);
    p.width_ema_alpha = d("width_ema_alpha", p.width_ema_alpha);
    p.centerline_ema_alpha = d("centerline_ema_alpha", p.centerline_ema_alpha);
    p.width_update_max_x = d("width_update_max_x", p.width_update_max_x);
    p.resample_step = d("resample_step", p.resample_step);
    p.position_window_m = d("position_window_m", p.position_window_m);
    p.tangent_window_m = d("tangent_window_m", p.tangent_window_m);
    p.far_near_field_range = d("far_near_field_range", p.far_near_field_range);
    p.lane_change_bezier = declare_parameter<bool>("lane_change_bezier", p.lane_change_bezier);
    p.lane_change_factor = d("lane_change_factor", p.lane_change_factor);
    p.lane_change_min_len = d("lane_change_min_len", p.lane_change_min_len);
    p.bezier_handle = d("bezier_handle", p.bezier_handle);
    return p;
  }

  // The clouds arrive on two topics; pair them by header stamp. Callbacks run on one thread
  // (default single-threaded executor), so no locking is needed.
  void onWhite(PointCloud2::ConstSharedPtr msg) {
    // A previous white cloud that never got its yellow partner (yellow dropped): process it alone
    // rather than silently losing the frame.
    if (white_ && !sameStamp(white_, msg)) {
      process(white_, nullptr);
      white_.reset();
    }
    white_ = std::move(msg);
    trySync();
  }

  void onYellow(PointCloud2::ConstSharedPtr msg) {
    yellow_ = std::move(msg);
    trySync();
  }

  void trySync() {
    if (!white_ || !yellow_) return;
    if (sameStamp(white_, yellow_)) {
      process(white_, yellow_);
      white_.reset();
      yellow_.reset();
    } else if (rclcpp::Time(yellow_->header.stamp) < rclcpp::Time(white_->header.stamp)) {
      yellow_.reset();  // stale yellow whose white partner was lost
    }
    // else: yellow is newer than white -> white is stale; it is flushed by the next onWhite().
  }

  static bool sameStamp(const PointCloud2::ConstSharedPtr& a, const PointCloud2::ConstSharedPtr& b) {
    return a->header.stamp.sec == b->header.stamp.sec && a->header.stamp.nanosec == b->header.stamp.nanosec;
  }

  void process(const PointCloud2::ConstSharedPtr& white, const PointCloud2::ConstSharedPtr& yellow) {
    const auto t0 = std::chrono::steady_clock::now();

    centerline::Result res;
    try {
      res = gen_->generate(toPoints(white), toPoints(yellow));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Centerline generation failed: %s", e.what());
    }

    // Always publish both (empty when a side is missing), like the Python node did.
    left_pub_->publish(toPathMsg(res.left, white->header));
    right_pub_->publish(toPathMsg(res.right, white->header));

    if (timing_log_period_ > 0) {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      sum_ms_ += ms;
      max_ms_ = std::max(max_ms_, ms);
      if (++frames_ >= timing_log_period_) {
        RCLCPP_INFO(get_logger(), "centerline: avg %.3f ms, max %.3f ms over %d frames", sum_ms_ / frames_,
                    max_ms_, frames_);
        sum_ms_ = max_ms_ = 0.0;
        frames_ = 0;
      }
    }
  }

  std::unique_ptr<centerline::CenterlineGenerator> gen_;
  rclcpp::Subscription<PointCloud2>::SharedPtr white_sub_, yellow_sub_;
  rclcpp::Publisher<Path>::SharedPtr left_pub_, right_pub_;
  PointCloud2::ConstSharedPtr white_, yellow_;

  int timing_log_period_ = 0;
  int frames_ = 0;
  double sum_ms_ = 0.0, max_ms_ = 0.0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CenterlineNode>());
  rclcpp::shutdown();
  return 0;
}