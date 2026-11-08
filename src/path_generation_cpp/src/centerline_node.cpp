// Generate robot-frame paths from synchronized accumulated skeleton maps.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "path_generation_cpp/skeleton_map.hpp"

#include "path_generation_cpp/centerline_generator.hpp"

using nav_msgs::msg::Path;
using Grid = nav_msgs::msg::OccupancyGrid;
using SyncPolicy = message_filters::sync_policies::ExactTime<Grid, Grid>;

namespace {

double yaw(const geometry_msgs::msg::Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

std::vector<centerline::Point2> toPoints(const Grid& grid, const centerline::RobotPose& robot,
                                       double max_x, double max_y) {
  const auto& info = grid.info;
  return centerline::skeletonMapPoints(grid.data,
      {static_cast<int>(info.width), static_cast<int>(info.height), info.resolution,
       info.origin.position.x, info.origin.position.y, yaw(info.origin.orientation)},
      robot, max_x, max_y);
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
    const auto white_topic = declare_parameter<std::string>("white_skeleton_topic", "/lane_map/white_skeleton");
    const auto yellow_topic = declare_parameter<std::string>("yellow_skeleton_topic", "/lane_map/yellow_skeleton");
    const auto left_topic = declare_parameter<std::string>("left_centerline_topic", "/path/left_centerline");
    const auto right_topic = declare_parameter<std::string>("right_centerline_topic", "/path/right_centerline");
    timing_log_period_ = static_cast<int>(declare_parameter<int64_t>("timing_log_period", 0));

    gen_ = std::make_unique<centerline::CenterlineGenerator>(declareGeneratorParams());

    base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
    max_lateral_range_ = declare_parameter<double>("max_lateral_range", 12.0);
    if (max_lateral_range_ <= 0) throw std::invalid_argument("max_lateral_range must be positive");
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    const auto qos = rclcpp::QoS(5).reliable().transient_local().get_rmw_qos_profile();
    white_sub_.subscribe(this, white_topic, qos);
    yellow_sub_.subscribe(this, yellow_topic, qos);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), white_sub_, yellow_sub_);
    sync_->registerCallback(std::bind(&CenterlineNode::process, this,
                                     std::placeholders::_1, std::placeholders::_2));

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

  void process(const Grid::ConstSharedPtr& white, const Grid::ConstSharedPtr& yellow) {
    const auto t0 = std::chrono::steady_clock::now();

    centerline::Result res;
    std_msgs::msg::Header header = white->header;
    header.frame_id = base_frame_;
    try {
      if (white->header.frame_id.empty() || white->header.frame_id != yellow->header.frame_id)
        throw std::runtime_error("Skeleton maps must share a nonempty frame");
      // The mapper already waited for this acquisition-time TF. Do not fall back
      // to a newer robot pose while retaining the older stamp on the paths.
      const auto tf = tf_buffer_->lookupTransform(white->header.frame_id, base_frame_,
          tf2_ros::fromMsg(white->header.stamp), tf2::durationFromSec(0.1));
      const centerline::RobotPose robot{tf.transform.translation.x, tf.transform.translation.y,
                                       yaw(tf.transform.rotation)};
      // Persistence now lives in odom-frame maps. Blending old robot-frame paths
      // would smear them as the vehicle moves, so only spatial smoothing is retained.
      gen_->resetMemory();
      res = gen_->generate(toPoints(*white, robot, gen_->params().max_range, max_lateral_range_),
                           toPoints(*yellow, robot, gen_->params().max_range, max_lateral_range_));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Map centerline generation failed: %s", e.what());
    }
    // Empty maps or unavailable TF clear both outputs instead of reusing an old path.
    left_pub_->publish(toPathMsg(res.left, header));
    right_pub_->publish(toPathMsg(res.right, header));

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
  message_filters::Subscriber<Grid> white_sub_, yellow_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string base_frame_;
  double max_lateral_range_;
  rclcpp::Publisher<Path>::SharedPtr left_pub_, right_pub_;

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