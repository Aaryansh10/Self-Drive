#pragma once
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>

#include "vision_cpp/cuda_utils.hpp"
#include "vision_cpp/gpu_kernels.hpp"
#include "vision_cpp/trt_engine.hpp"
#include "vision_cpp/types.hpp"

namespace vision_cpp {

struct PipelineConfig {
  int in_w = 960, in_h = 544;            // model input == mask resolution
  int subsample_step = 1;
  int open_k = 3, close_k = 5;           // mask cleaning kernels (odd; <=1 disables)
  int min_area = 150, min_perimeter = 50;
  float score_thresh = 0.35f;
  std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
  std::array<float, 3> stdv{0.229f, 0.224f, 0.225f};
  bool use_graph = true;
};

// Whole per-frame GPU workload as ONE CUDA graph (one launch per frame):
//   H2D(frame) -> preprocess -> TensorRT -> detection candidates
//   -> argmax -> open/close -> union-find CC + area/perimeter filter -> stop-line edge
//   -> D2H(cleaned white/yellow masks + FrameResults)
// All buffers are allocated in the constructor / set_source(); nothing is allocated per frame.
class GpuPipeline {
 public:
  GpuPipeline(const PipelineConfig& cfg, TrtEngine& engine, TrtEngine::LogFn log);
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Declares the camera frame layout. Cheap no-op when unchanged; otherwise re-sizes the staging
  // buffer and re-captures the graph (only happens on the first frame or if the camera changes).
  void set_source(int w, int h, int stride, int channels, bool bgr);

  uint8_t* staging() { return h_src_.get(); }  // pinned; fill with stride*h bytes, then launch()
  void launch();                               // async: one graph launch
  void wait();                                 // block until results are valid
  const FrameResults& results() const { return *h_res_.get(); }
  float gpu_ms() const { return gpu_ms_; }
  bool graph_active() const { return graph_exec_ != nullptr; }

  const uint8_t* cleaned_masks() const { return h_masks_.get(); }
  int cols() const { return cols_; }

 private:
  void enqueue_all(cudaStream_t s);
  void capture();

  PipelineConfig cfg_;
  TrtEngine& engine_;
  TrtEngine::LogFn log_;
  int cols_ = 0;

  // source frame description
  int sw_ = 0, sh_ = 0, stride_ = 0, sc_ = 3;
  bool bgr_ = true;
  size_t src_bytes_ = 0;
  bool configured_ = false;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_start_ = nullptr, ev_end_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  float gpu_ms_ = 0.f;

  PinnedBuffer<uint8_t> h_src_;
  PinnedBuffer<FrameResults> h_res_;
  DeviceBuffer<uint8_t> d_src_;
  DeviceBuffer<FrameResults> d_res_;
  DeviceBuffer<uint8_t> planes_a_, planes_b_;   // 3 x H x W each
  DeviceBuffer<int> cc_parent_;                 // 3 x H x W
  DeviceBuffer<int2> cc_stats_;                 // 3 x H x W
  PinnedBuffer<uint8_t> h_masks_;  // two full-resolution cleaned lane masks
};

}  // namespace vision_cpp
