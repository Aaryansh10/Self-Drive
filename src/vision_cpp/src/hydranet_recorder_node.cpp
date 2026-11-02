// Port of model_output.py: run HydraNet (TensorRT) on a camera stream, draw detections and the
// segmentation overlay, record to a video file and republish the overlay.
//
// GPU: preprocess, TensorRT, detection thresholding, segmentation argmax + alpha blending.
// CPU: NMS on a few dozen boxes, box/label drawing, video encoding.
//
// Note: the original python file contains a paste accident in its parameter block
// (`p("num_objweights_path = ..._classes")`); here num_obj_classes is simply read from the engine.
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "vision_cpp/cuda_utils.hpp"
#include "vision_cpp/gpu_kernels.hpp"
#include "vision_cpp/model_helpers.hpp"
#include "vision_cpp/trt_engine.hpp"

using namespace vision_cpp;

class HydraNetRecorder : public rclcpp::Node {
 public:
  HydraNetRecorder() : rclcpp::Node("hydranet_recorder") {
    declare_parameter("image_topic", "/zed_node/monocamera/image_raw");
    declare_parameter("output_topic", "/hydranet/overlay");  // "" disables
    declare_parameter("output_path", "hydranet_output.mp4");
    declare_parameter("fps", 30.0);
    declare_parameter("fourcc", "mp4v");
    declare_parameter("onnx_path", "hydranet.onnx");
    declare_parameter("engine_path", "");
    declare_parameter("use_fp16", true);
    declare_parameter("input_height", 544);
    declare_parameter("input_width", 960);
    declare_parameter("class_names", std::vector<std::string>{});
    declare_parameter("score_thresh", 0.35);
    declare_parameter("nms_iou", 0.5);
    declare_parameter("max_dets", 100);
    declare_parameter("mask_alpha", 0.45);
    declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
    declare_parameter("std", std::vector<double>{0.229, 0.224, 0.225});

    auto P = [this](const char* n) { return get_parameter(n); };
    in_h_ = static_cast<int>(P("input_height").as_int());
    in_w_ = static_cast<int>(P("input_width").as_int());
    thr_ = static_cast<float>(P("score_thresh").as_double());
    nms_iou_ = static_cast<float>(P("nms_iou").as_double());
    max_dets_ = static_cast<int>(P("max_dets").as_int());
    alpha_ = static_cast<float>(P("mask_alpha").as_double());
    fps_ = P("fps").as_double();
    fourcc_ = P("fourcc").as_string();
    out_path_ = P("output_path").as_string();
    const auto mean = P("mean").as_double_array(), stdv = P("std").as_double_array();
    for (int i = 0; i < 3; ++i) {
      mean_[i] = static_cast<float>(mean.at(i));
      std_[i] = static_cast<float>(stdv.at(i));
    }

    auto log = [this](const std::string& s) { RCLCPP_INFO(get_logger(), "%s", s.c_str()); };
    TrtEngine::Options eo;
    eo.onnx_path = P("onnx_path").as_string();
    eo.engine_path = P("engine_path").as_string();
    eo.fp16 = P("use_fp16").as_bool();
    eo.in_h = in_h_;
    eo.in_w = in_w_;
    engine_ = std::make_unique<TrtEngine>(eo, log);

    const int K = engine_->n_classes();
    for (const auto& n : P("class_names").as_string_array())
      if (!n.empty()) names_.push_back(n);
    if (static_cast<int>(names_.size()) != K) {
      names_.clear();
      for (int i = 0; i < K; ++i) names_.push_back("cls" + std::to_string(i));
    }

    VCUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    d_res_.alloc(1);
    h_res_.alloc(1);
    for (int i = 0; i < 3; ++i) engine_->enqueue(stream_);  // warm-up
    VCUDA(cudaStreamSynchronize(stream_));

    rclcpp::QoS qos = rclcpp::SensorDataQoS();
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        P("image_topic").as_string(), qos, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { image_cb(std::move(m)); });
    const std::string out_topic = P("output_topic").as_string();
    if (!out_topic.empty()) pub_ = create_publisher<sensor_msgs::msg::Image>(out_topic, 1);
    RCLCPP_INFO(get_logger(), "Recording '%s' -> %s", P("image_topic").as_string().c_str(), out_path_.c_str());
  }

  ~HydraNetRecorder() override {
    if (stream_) cudaStreamSynchronize(stream_);
    close();
    if (stream_) cudaStreamDestroy(stream_);
  }

  void close() {
    if (writer_.isOpened()) {
      writer_.release();
      RCLCPP_INFO(get_logger(), "Saved %d frames to %s", frames_, out_path_.c_str());
    }
  }

 private:
  void setup_frame(int w, int h, int stride) {
    const size_t in_bytes = static_cast<size_t>(stride) * h, out_bytes = static_cast<size_t>(w) * h * 3;
    h_src_.alloc(in_bytes);
    d_src_.alloc(in_bytes);
    d_out_.alloc(out_bytes);
    h_out_.alloc(out_bytes);
    fw_ = w;
    fh_ = h;
    fstride_ = stride;
    msg_.width = w;
    msg_.height = h;
    msg_.encoding = "bgr8";
    msg_.step = w * 3;
    msg_.data.resize(out_bytes);
  }

  void image_cb(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    const std::string& e = msg->encoding;
    int ch = 3;
    bool bgr = true;
    if (e == "bgr8") { ch = 3; bgr = true; }
    else if (e == "rgb8") { ch = 3; bgr = false; }
    else if (e == "bgra8") { ch = 4; bgr = true; }
    else if (e == "rgba8") { ch = 4; bgr = false; }
    else { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported encoding %s", e.c_str()); return; }

    const int w = static_cast<int>(msg->width), h = static_cast<int>(msg->height), stride = static_cast<int>(msg->step);
    if (w != fw_ || h != fh_ || stride != fstride_) {
      if (writer_.isOpened()) { RCLCPP_ERROR(get_logger(), "frame size changed while recording"); return; }
      setup_frame(w, h, stride);
    }
    std::memcpy(h_src_.get(), msg->data.data(), static_cast<size_t>(stride) * h);

    const cudaStream_t s = stream_;
    VCUDA(cudaMemsetAsync(d_res_.get(), 0, 4 * sizeof(int), s));
    VCUDA(cudaMemcpyAsync(d_src_.get(), h_src_.get(), static_cast<size_t>(stride) * h, cudaMemcpyHostToDevice, s));
    launch_preprocess(d_src_.get(), w, h, stride, ch, bgr, engine_->input_ptr(), engine_->input_half(), in_w_, in_h_, mean_, std_, s);
    if (!engine_->enqueue(s)) { RCLCPP_ERROR(get_logger(), "TensorRT enqueue failed"); return; }
    launch_det_candidates(engine_->cls(), engine_->ctr(), engine_->boxes(), engine_->n_anchors(), engine_->n_classes(), thr_, d_res_.get(), s);
    launch_overlay(d_src_.get(), w, h, stride, ch, bgr, engine_->lane(), engine_->lane_c(), engine_->lane_h(), engine_->lane_w(), alpha_, d_out_.get(), s);
    VCUDA(cudaMemcpyAsync(h_out_.get(), d_out_.get(), static_cast<size_t>(w) * h * 3, cudaMemcpyDeviceToHost, s));
    VCUDA(cudaMemcpyAsync(h_res_.get(), d_res_.get(), offsetof(FrameResults, skel), cudaMemcpyDeviceToHost, s));  // counters + candidates only
    VCUDA(cudaStreamSynchronize(s));

    select_detections(*h_res_.get(), nms_iou_, max_dets_, static_cast<float>(w) / in_w_, static_cast<float>(h) / in_h_, w, h, -1.f, scratch_, dets_);

    cv::Mat out(h, w, CV_8UC3, h_out_.get());
    for (const auto& d : dets_) {
      const cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1)), p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
      cv::rectangle(out, p1, p2, cv::Scalar(0, 255, 255), 2);
      char buf[96];
      std::snprintf(buf, sizeof buf, "%s %.2f", names_[static_cast<size_t>(d.label)].c_str(), d.score);
      int bl = 0;
      const cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
      const int ty = std::max(p1.y, ts.height + bl + 2);
      cv::rectangle(out, cv::Point(p1.x, ty - ts.height - bl - 2), cv::Point(p1.x + ts.width + 2, ty), cv::Scalar(0, 255, 255), -1);
      cv::putText(out, buf, cv::Point(p1.x + 1, ty - bl - 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    if (!writer_.isOpened()) {  // lazily, once the real frame size is known
      writer_.open(out_path_, cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]), fps_, cv::Size(w, h));
      if (!writer_.isOpened()) { RCLCPP_ERROR(get_logger(), "Could not open VideoWriter for %s", out_path_.c_str()); throw std::runtime_error("VideoWriter failed to open"); }
    }
    writer_.write(out);

    if (pub_) {
      msg_.header = msg->header;
      std::memcpy(msg_.data.data(), h_out_.get(), msg_.data.size());
      pub_->publish(msg_);
    }
    ++frames_;
  }

  int in_h_, in_w_, max_dets_, frames_ = 0, fw_ = 0, fh_ = 0, fstride_ = 0;
  float thr_, nms_iou_, alpha_, mean_[3], std_[3];
  double fps_;
  std::string fourcc_, out_path_;
  std::vector<std::string> names_;

  std::unique_ptr<TrtEngine> engine_;
  cudaStream_t stream_ = nullptr;
  PinnedBuffer<uint8_t> h_src_, h_out_;
  DeviceBuffer<uint8_t> d_src_, d_out_;
  PinnedBuffer<FrameResults> h_res_;
  DeviceBuffer<FrameResults> d_res_;
  DetScratch scratch_;
  std::vector<Detection> dets_;

  cv::VideoWriter writer_;
  sensor_msgs::msg::Image msg_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HydraNetRecorder>();
  try {
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "%s", e.what());
  }
  node->close();
  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}
