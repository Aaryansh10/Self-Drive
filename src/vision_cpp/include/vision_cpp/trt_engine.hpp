#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>

#include "vision_cpp/cuda_utils.hpp"
#include "vision_cpp/types.hpp"

namespace vision_cpp {

// Loads (or builds from ONNX and caches) a static-shape TensorRT engine and owns ALL its IO
// buffers. Addresses are bound once; enqueue() allocates nothing, so it can live in a CUDA graph.
//
// Expected IO (batch 1, static):
//   image       (1,3,H,W)   normalised RGB
//   lane_logits (1,C,H,W)   segmentation logits (C >= 4)
//   cls_logits  (1,N,K)     detection class logits
//   centerness  (1,N,1)     centerness logits
//   boxes       (1,N,4)     xyxy in input pixels
class TrtEngine {
 public:
  using LogFn = std::function<void(const std::string&)>;

  struct Options {
    std::string onnx_path;
    std::string engine_path;  // "" -> <onnx without extension>.<fp16|fp32>.engine
    bool fp16 = true;
    bool rebuild = false;     // force re-build even if a cached engine exists
    int workspace_mb = 2048;
    int opt_level = 3;        // TensorRT >= 8.6 builder optimisation level (0-5)
    int in_h = 544, in_w = 960;
    std::string input_name = "image", lane_name = "lane_logits", cls_name = "cls_logits",
                ctr_name = "centerness", box_name = "boxes";
  };

  TrtEngine(const Options& opt, LogFn log);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // ONNX -> serialised engine on disk. Used by the constructor and by the build_engine tool.
  static bool build(const Options& opt, const LogFn& log);
  static std::string default_engine_path(const Options& opt);

  bool enqueue(cudaStream_t s) { return ctx_->enqueueV3(s); }

  void* input_ptr() const { return io_[0].get(); }
  bool input_half() const { return half_[0]; }
  TensorRef lane() const { return {io_[1].get(), half_[1]}; }
  TensorRef cls() const { return {io_[2].get(), half_[2]}; }
  TensorRef ctr() const { return {io_[3].get(), half_[3]}; }
  TensorRef boxes() const { return {io_[4].get(), half_[4]}; }

  int lane_c() const { return lane_c_; }
  int lane_h() const { return lane_h_; }
  int lane_w() const { return lane_w_; }
  int n_anchors() const { return n_anchors_; }
  int n_classes() const { return n_classes_; }

 private:
  bool try_load(const std::string& path);
  bool validate();
  void allocate_io();

  Options opt_;
  LogFn log_;
  std::unique_ptr<nvinfer1::ILogger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;

  DeviceBuffer<uint8_t> io_[5];
  bool half_[5] = {false, false, false, false, false};
  int lane_c_ = 0, lane_h_ = 0, lane_w_ = 0, n_anchors_ = 0, n_classes_ = 0;
};

}  // namespace vision_cpp
