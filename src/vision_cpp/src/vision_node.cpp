#include "vision_cpp/vision_node.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <sensor_msgs/msg/point_field.hpp>
#include <vision_interfaces/msg/detection2_d.hpp>

namespace vision_cpp {
namespace {

// Supported camera encodings. Everything else is rejected (cv_bridge is not needed at all).
bool parse_encoding(const std::string& enc, int& channels, bool& bgr) {
  if (enc == "bgr8") { channels = 3; bgr = true; return true; }
  if (enc == "rgb8") { channels = 3; bgr = false; return true; }
  if (enc == "bgra8") { channels = 4; bgr = true; return true; }
  if (enc == "rgba8") { channels = 4; bgr = false; return true; }
  return false;
}

inline double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

VisionNode::VisionNode() : rclcpp::Node("vision") {
  // ---- parameters (same names/defaults as the python node where they still make sense) --------
  declare_parameter("image_topic", "/zed_node/monocamera/image_raw");
  declare_parameter("cam_info_topic", "/zed_node/monocamera/camera_info");
  declare_parameter("imu_topic", "/imu");
  declare_parameter("pc_whitelane_topic", "/pc_whitelane");
  declare_parameter("pc_yellowlane_topic", "/pc_yellowlane");
  declare_parameter("pc_stopline_topic", "/pc_stopline");
  declare_parameter("detections_topic", "/detections");
  declare_parameter("ground_frame", "link_base");

  declare_parameter("subsample_step", 1);
  declare_parameter("cam_pitch", 0.4);
  declare_parameter("cam_height", 1.606);
  declare_parameter("max_range", 8.0);

  declare_parameter("onnx_path", "hydranet.onnx");
  declare_parameter("engine_path", "");        // "" -> next to the onnx file
  declare_parameter("rebuild_engine", false);
  declare_parameter("use_fp16", true);
  declare_parameter("trt_workspace_mb", 2048);
  declare_parameter("trt_opt_level", 3);
  declare_parameter("input_height", 544);
  declare_parameter("input_width", 960);
  declare_parameter("num_seg_classes", 4);
  declare_parameter("num_obj_classes", 5);
  declare_parameter("class_names", std::vector<std::string>{"barrel", "tyre", "pothole", "sign", "pedestrian"});
  declare_parameter("score_thresh", 0.35);
  declare_parameter("nms_iou", 0.5);
  declare_parameter("max_detections", 50);
  declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
  declare_parameter("std", std::vector<double>{0.229, 0.224, 0.225});

  declare_parameter("open_k", 3);
  declare_parameter("close_k", 5);
  declare_parameter("min_area", 150);
  declare_parameter("min_perimeter", 50);
  declare_parameter("thin_row_start", 160);
  declare_parameter("thin_max_iter", 64);
  declare_parameter("cuda_graph", true);

  declare_parameter("spin_wait", true);        // busy-wait on the GPU: lowest latency, burns one core
  declare_parameter("lock_memory", true);      // mlockall: no page faults in the hot path
  declare_parameter("rt_priority", 0);         // SCHED_FIFO priority for the image thread (0 = off)
  declare_parameter("profile_timing", true);
  declare_parameter("profile_period", 50);

  auto P = [this](const char* n) { return get_parameter(n); };
  ground_frame_ = P("ground_frame").as_string();
  step_ = static_cast<int>(P("subsample_step").as_int());
  cam_pitch_ = P("cam_pitch").as_double();
  cam_height_ = P("cam_height").as_double();
  max_range_ = P("max_range").as_double();
  in_h_ = static_cast<int>(P("input_height").as_int());
  in_w_ = static_cast<int>(P("input_width").as_int());
  nms_iou_ = static_cast<float>(P("nms_iou").as_double());
  max_dets_ = static_cast<int>(P("max_detections").as_int());
  class_names_ = P("class_names").as_string_array();
  profile_ = P("profile_timing").as_bool();
  profile_period_ = std::max<int>(1, static_cast<int>(P("profile_period").as_int()));
  const int num_obj = static_cast<int>(P("num_obj_classes").as_int());

  auto log = [this](const std::string& s) { RCLCPP_INFO(get_logger(), "%s", s.c_str()); };

  // Must happen before the CUDA context is created (i.e. before TensorRT is touched).
  if (P("spin_wait").as_bool()) {
    const cudaError_t e = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if (e != cudaSuccess) RCLCPP_WARN(get_logger(), "cudaSetDeviceFlags(spin): %s", cudaGetErrorString(e));
  }
  if (P("lock_memory").as_bool() && mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
    RCLCPP_WARN(get_logger(), "mlockall failed (raise RLIMIT_MEMLOCK to enable)");

  // ---- model + GPU pipeline ---------------------------------------------------------------------
  TrtEngine::Options eo;
  eo.onnx_path = P("onnx_path").as_string();
  eo.engine_path = P("engine_path").as_string();
  eo.rebuild = P("rebuild_engine").as_bool();
  eo.fp16 = P("use_fp16").as_bool();
  eo.workspace_mb = static_cast<int>(P("trt_workspace_mb").as_int());
  eo.opt_level = static_cast<int>(P("trt_opt_level").as_int());
  eo.in_h = in_h_;
  eo.in_w = in_w_;
  engine_ = std::make_unique<TrtEngine>(eo, log);

  if (engine_->n_classes() != num_obj)
    RCLCPP_WARN(get_logger(), "engine has %d detection classes but num_obj_classes=%d", engine_->n_classes(), num_obj);
  if (static_cast<int>(class_names_.size()) != engine_->n_classes()) {
    class_names_.clear();
    for (int i = 0; i < engine_->n_classes(); ++i) class_names_.push_back("cls" + std::to_string(i));
  }

  PipelineConfig pc;
  pc.in_w = in_w_;
  pc.in_h = in_h_;
  pc.subsample_step = step_;
  pc.thin_row_start = static_cast<int>(P("thin_row_start").as_int());
  pc.open_k = static_cast<int>(P("open_k").as_int());
  pc.close_k = static_cast<int>(P("close_k").as_int());
  pc.min_area = static_cast<int>(P("min_area").as_int());
  pc.min_perimeter = static_cast<int>(P("min_perimeter").as_int());
  pc.thin_max_iter = static_cast<int>(P("thin_max_iter").as_int());
  pc.score_thresh = static_cast<float>(P("score_thresh").as_double());
  pc.use_graph = P("cuda_graph").as_bool();
  const auto mean = P("mean").as_double_array(), stdv = P("std").as_double_array();
  for (int i = 0; i < 3; ++i) {
    pc.mean[i] = static_cast<float>(mean.at(i));
    pc.stdv[i] = static_cast<float>(stdv.at(i));
  }
  pipe_ = std::make_unique<GpuPipeline>(pc, *engine_, log);
  pipe_->set_source(in_w_, in_h_, in_w_ * 3, 3, true);  // warm everything for the expected format

  // ---- persistent messages / scratch ------------------------------------------------------------
  for (auto* m : {&pc_white_, &pc_yellow_, &pc_stop_}) {
    init_cloud(*m);
    m->data.reserve(12 * static_cast<size_t>(std::max(kMaxSkel, kMaxCols)));
  }
  pts_.assign(3 * static_cast<size_t>(std::max(kMaxSkel, kMaxCols)), 0.f);
  det_msg_.detections.reserve(static_cast<size_t>(max_dets_));
  for (auto* v : {&vw_, &uw_, &vy_, &uy_}) v->reserve(kMaxSkel);
  for (auto* v : {&vs_, &us_}) v->reserve(kMaxCols);
  dets_.reserve(static_cast<size_t>(max_dets_) + 8);
  det_scratch_.idx.reserve(kMaxCand);
  det_scratch_.keep.reserve(kMaxCand);

  // ---- ROS interfaces ---------------------------------------------------------------------------
  // The image callback gets its own mutually-exclusive group and thread, so IMU / camera-info are
  // never blocked behind a frame and the pitch used by the IPM is always the freshest one.
  cb_img_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cb_aux_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions o_img, o_aux;
  o_img.callback_group = cb_img_;
  o_aux.callback_group = cb_aux_;

  // depth=1 + best effort: always process the newest frame, never queue stale ones.
  rclcpp::QoS image_qos(rclcpp::KeepLast(1));
  image_qos.best_effort();
  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      P("image_topic").as_string(), image_qos, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { img_cb(std::move(m)); }, o_img);
  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      P("cam_info_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { cam_info_cb(std::move(m)); }, o_aux);
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      P("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { imu_cb(std::move(m)); }, o_aux);

  pub_white_ = create_publisher<sensor_msgs::msg::PointCloud2>(P("pc_whitelane_topic").as_string(), rclcpp::SensorDataQoS());
  pub_yellow_ = create_publisher<sensor_msgs::msg::PointCloud2>(P("pc_yellowlane_topic").as_string(), rclcpp::SensorDataQoS());
  pub_stop_ = create_publisher<sensor_msgs::msg::PointCloud2>(P("pc_stopline_topic").as_string(), rclcpp::SensorDataQoS());
  pub_det_ = create_publisher<vision_interfaces::msg::DetectionArray>(
      P("detections_topic").as_string(), rclcpp::QoS(rclcpp::KeepLast(5)).reliable());

  RCLCPP_INFO(get_logger(), "Vision node started successfully.");
}

// -------------------------------------------------------------------------------------------------
void VisionNode::init_cloud(sensor_msgs::msg::PointCloud2& pc) const {
  using sensor_msgs::msg::PointField;
  pc.height = 1;
  pc.is_bigendian = false;
  pc.is_dense = true;
  pc.point_step = 12;
  pc.fields.resize(3);
  const char* names[3] = {"x", "y", "z"};
  for (uint32_t i = 0; i < 3; ++i) {
    pc.fields[i].name = names[i];
    pc.fields[i].offset = 4 * i;
    pc.fields[i].datatype = PointField::FLOAT32;
    pc.fields[i].count = 1;
  }
}

void VisionNode::fill_cloud(sensor_msgs::msg::PointCloud2& pc, const std_msgs::msg::Header& h, size_t n) {
  pc.header = h;
  pc.width = static_cast<uint32_t>(n);
  pc.row_step = static_cast<uint32_t>(12 * n);
  pc.data.resize(12 * n);
  if (n) std::memcpy(pc.data.data(), pts_.data(), 12 * n);
}

// -------------------------------------------------------------------------------------------------
void VisionNode::img_cb(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!cam_ready_.load(std::memory_order_acquire)) return;

  static thread_local bool rt_done = false;
  if (!rt_done) {
    rt_done = true;
    const int prio = static_cast<int>(get_parameter("rt_priority").as_int());
    if (prio > 0) {
      sched_param sp{};
      sp.sched_priority = prio;
      if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
        RCLCPP_WARN(get_logger(), "could not set SCHED_FIFO %d (need CAP_SYS_NICE / rtprio limit)", prio);
    }
  }

  const auto t0 = Clock::now();
  int ch = 0;
  bool bgr = true;
  if (!parse_encoding(msg->encoding, ch, bgr)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported encoding '%s' (bgr8/rgb8/bgra8/rgba8)", msg->encoding.c_str());
    return;
  }
  const int w = static_cast<int>(msg->width), h = static_cast<int>(msg->height);
  const int stride = static_cast<int>(msg->step);
  if (stride < w * ch || msg->data.size() < static_cast<size_t>(stride) * h) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "malformed image message");
    return;
  }

  std_msgs::msg::Header header;
  header.stamp = msg->header.stamp;
  header.frame_id = ground_frame_;

  // ---- upload + GPU (one graph launch) ----------------------------------------------------------
  pipe_->set_source(w, h, stride, ch, bgr);  // no-op in steady state
  std::memcpy(pipe_->staging(), msg->data.data(), static_cast<size_t>(stride) * h);
  const auto t1 = Clock::now();
  pipe_->launch();
  pipe_->wait();
  const auto t2 = Clock::now();

  // ---- CPU tail: sort / NMS / IPM ---------------------------------------------------------------
  const FrameResults& r = pipe_->results();
  const int r0 = pipe_->thin_r0(), rows = pipe_->thin_rows();

  select_detections(r, nms_iou_, max_dets_, 1.f, 1.f, in_w_, in_h_, 2.f, det_scratch_, dets_);

  const int nw = std::min(r.skel_count[0], kMaxSkel), ny = std::min(r.skel_count[1], kMaxSkel);
  if (r.skel_count[0] > kMaxSkel || r.skel_count[1] > kMaxSkel)
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "skeleton pixel list overflow (raise kMaxSkel)");
  order_skeleton_pixels(r.skel[0], nw, r0, rows, skel_scratch_w_, vw_, uw_);
  order_skeleton_pixels(r.skel[1], ny, r0, rows, skel_scratch_y_, vy_, uy_);

  vs_.clear();
  us_.clear();
  for (int u = 0; u < pipe_->cols(); ++u)
    if (r.stop_v[u] >= 0) {
      vs_.push_back(r.stop_v[u]);
      us_.push_back(u);
    }

  const double pitch = pitch_bot_.load(std::memory_order_relaxed);
  size_t n = ipm_.project(vw_.data(), uw_.data(), vw_.size(), pitch, pts_.data());
  fill_cloud(pc_white_, header, n);
  n = ipm_.project(vy_.data(), uy_.data(), vy_.size(), pitch, pts_.data());
  fill_cloud(pc_yellow_, header, n);
  n = ipm_.project(vs_.data(), us_.data(), vs_.size(), pitch, pts_.data());
  fill_cloud(pc_stop_, header, n);

  det_msg_.header = header;
  det_msg_.detections.resize(dets_.size());
  for (size_t i = 0; i < dets_.size(); ++i) {
    auto& d = det_msg_.detections[i];
    d.class_id = dets_[i].label;
    d.class_name = class_names_[static_cast<size_t>(dets_[i].label)];
    d.score = dets_[i].score;
    d.x1 = dets_[i].x1;
    d.y1 = dets_[i].y1;
    d.x2 = dets_[i].x2;
    d.y2 = dets_[i].y2;
  }
  const auto t3 = Clock::now();

  pub_white_->publish(pc_white_);
  pub_yellow_->publish(pc_yellow_);
  pub_stop_->publish(pc_stop_);
  pub_det_->publish(det_msg_);
  const auto t4 = Clock::now();

  if (profile_) {
    acc_copy_ += ms(t0, t1);
    acc_gpu_ += pipe_->gpu_ms();
    acc_wait_ += ms(t1, t2);
    acc_cpu_ += ms(t2, t3);
    acc_pub_ += ms(t3, t4);
    const double total = ms(t0, t4);
    acc_total_ += total;
    max_total_ = std::max(max_total_, total);
    acc_lat_ += (now() - rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type())).seconds() * 1e3;
    if (++prof_frames_ >= profile_period_) report_profile();
  }
}

void VisionNode::report_profile() {
  const auto now_t = Clock::now();
  const double secs = std::chrono::duration<double>(now_t - last_report_).count();
  const double k = 1.0 / prof_frames_;
  RCLCPP_INFO(get_logger(),
              "[profile] %d frames, %.1f Hz | avg ms: copy %.2f | gpu %.2f (wait %.2f) | cpu %.2f | publish %.2f | TOTAL %.2f (max %.2f) | "
              "stamp->done latency %.2f",
              prof_frames_, prof_frames_ / secs, acc_copy_ * k, acc_gpu_ * k, acc_wait_ * k, acc_cpu_ * k, acc_pub_ * k,
              acc_total_ * k, max_total_, acc_lat_ * k);
  acc_copy_ = acc_gpu_ = acc_wait_ = acc_cpu_ = acc_pub_ = acc_total_ = acc_lat_ = max_total_ = 0.0;
  prof_frames_ = 0;
  last_report_ = now_t;
}

// -------------------------------------------------------------------------------------------------
void VisionNode::imu_cb(sensor_msgs::msg::Imu::ConstSharedPtr msg) {
  const auto& q = msg->orientation;
  const double sinp = 2.0 * (q.w * q.y - q.z * q.x);
  pitch_bot_.store(std::fabs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp),
                   std::memory_order_relaxed);
}

void VisionNode::cam_info_cb(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  if (cam_ready_.load()) return;
  // If the camera resolution differs from the model input the GPU resizes the frame, so the
  // intrinsics have to be scaled the same way.
  const double sx = msg->width > 0 ? static_cast<double>(in_w_) / msg->width : 1.0;
  const double sy = msg->height > 0 ? static_cast<double>(in_h_) / msg->height : 1.0;
  IpmParams p;
  p.fx = msg->k[0] * sx;
  p.cx = msg->k[2] * sx;
  p.fy = msg->k[4] * sy;
  p.cy = msg->k[5] * sy;
  p.cam_pitch = cam_pitch_;
  p.cam_height = cam_height_;
  p.max_range = max_range_;
  p.subsample_step = step_;
  p.full_h = in_h_;
  ipm_.init(p);
  cam_ready_.store(true, std::memory_order_release);
  RCLCPP_INFO(get_logger(), "Camera intrinsics received. fx:%.2f, fy:%.2f, cx:%.2f, cy:%.2f", p.fx, p.fy, p.cx, p.cy);
  cam_info_sub_.reset();
}

}  // namespace vision_cpp
