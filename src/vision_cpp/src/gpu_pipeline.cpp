#include "vision_cpp/gpu_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace vision_cpp {

GpuPipeline::GpuPipeline(const PipelineConfig& cfg, TrtEngine& engine, TrtEngine::LogFn log)
    : cfg_(cfg), engine_(engine), log_(std::move(log)) {
  if (engine_.lane_w() != cfg_.in_w || engine_.lane_h() != cfg_.in_h)
    throw std::runtime_error("lane_logits resolution must equal the model input resolution");
  if (engine_.lane_c() < 4) throw std::runtime_error("need >= 4 segmentation classes (bg + 3 lane classes)");
  cfg_.subsample_step = std::max(1, cfg_.subsample_step);
  cfg_.open_k |= (cfg_.open_k > 1);    // force odd kernel sizes
  cfg_.close_k |= (cfg_.close_k > 1);
  cfg_.thin_max_iter = std::min(std::max(2, cfg_.thin_max_iter), kThinMaxIter);

  const int s = cfg_.subsample_step;
  const int Hs = (cfg_.in_h + s - 1) / s, Ws = (cfg_.in_w + s - 1) / s;
  if (Ws > kMaxCols) throw std::runtime_error("image too wide for kMaxCols");
  geom_.W = cfg_.in_w;
  geom_.H = cfg_.in_h;
  geom_.step = s;
  geom_.r0 = std::min(std::max(0, (cfg_.thin_row_start + s - 1) / s), Hs - 1);
  geom_.Wt = Ws;
  geom_.Ht = Hs - geom_.r0;
  cols_ = Ws;

  const size_t hw = static_cast<size_t>(cfg_.in_w) * cfg_.in_h;
  const size_t tp = static_cast<size_t>(geom_.Wt) * geom_.Ht;
  planes_a_.alloc(3 * hw);
  planes_b_.alloc(3 * hw);
  cc_parent_.alloc(3 * hw);
  cc_stats_.alloc(3 * hw);
  thin0_.alloc(2 * tp);
  thin1_.alloc(2 * tp);
  thin_counters_.alloc(kThinMaxIter);
  skel_parent_.alloc(2 * tp);
  skel_stats_.alloc(2 * tp);
  d_res_.alloc(1);
  h_res_.alloc(1);

  int prio_lo = 0, prio_hi = 0;
  VCUDA(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  VCUDA(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, prio_hi));
  VCUDA(cudaEventCreate(&ev_start_));
  VCUDA(cudaEventCreate(&ev_end_));

  // Let TensorRT load its kernels / pick tactics before anything is captured.
  for (int i = 0; i < 3; ++i)
    if (!engine_.enqueue(stream_)) throw std::runtime_error("TensorRT warm-up enqueue failed");
  VCUDA(cudaStreamSynchronize(stream_));
}

GpuPipeline::~GpuPipeline() {
  if (stream_) cudaStreamSynchronize(stream_);
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  if (graph_) cudaGraphDestroy(graph_);
  if (ev_start_) cudaEventDestroy(ev_start_);
  if (ev_end_) cudaEventDestroy(ev_end_);
  if (stream_) cudaStreamDestroy(stream_);
}

void GpuPipeline::set_source(int w, int h, int stride, int channels, bool bgr) {
  if (configured_ && w == sw_ && h == sh_ && stride == stride_ && channels == sc_ && bgr == bgr_) return;
  VCUDA(cudaStreamSynchronize(stream_));
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  if (graph_) cudaGraphDestroy(graph_);
  graph_exec_ = nullptr;
  graph_ = nullptr;

  sw_ = w;
  sh_ = h;
  stride_ = stride;
  sc_ = channels;
  bgr_ = bgr;
  src_bytes_ = static_cast<size_t>(stride) * h;
  if (src_bytes_ > h_src_.size()) {
    h_src_.alloc(src_bytes_);
    d_src_.alloc(src_bytes_);
  }
  if (w != cfg_.in_w || h != cfg_.in_h)
    log_("[pipeline] camera " + std::to_string(w) + "x" + std::to_string(h) + " != model input " +
         std::to_string(cfg_.in_w) + "x" + std::to_string(cfg_.in_h) + ": resizing on the GPU");

  enqueue_all(stream_);  // eager warm-up of every kernel
  VCUDA(cudaStreamSynchronize(stream_));
  configured_ = true;
  if (cfg_.use_graph) capture();
  log_(std::string("[pipeline] ready, CUDA graph ") + (graph_exec_ ? "ON" : "OFF"));
}

void GpuPipeline::enqueue_all(cudaStream_t s) {
  const int W = cfg_.in_w, H = cfg_.in_h;
  const size_t hw = static_cast<size_t>(W) * H;

  VCUDA(cudaMemsetAsync(d_res_.get(), 0, 4 * sizeof(int), s));  // the 4 counters
  VCUDA(cudaMemsetAsync(thin_counters_.get(), 0, kThinMaxIter * sizeof(int), s));
  VCUDA(cudaMemcpyAsync(d_src_.get(), h_src_.get(), src_bytes_, cudaMemcpyHostToDevice, s));

  // --- model --------------------------------------------------------------------------------
  launch_preprocess(d_src_.get(), sw_, sh_, stride_, sc_, bgr_, engine_.input_ptr(),
                    engine_.input_half(), W, H, cfg_.mean.data(), cfg_.stdv.data(), s);
  if (!engine_.enqueue(s)) throw std::runtime_error("TensorRT enqueue failed");

  // --- detections ---------------------------------------------------------------------------
  launch_det_candidates(engine_.cls(), engine_.ctr(), engine_.boxes(), engine_.n_anchors(),
                        engine_.n_classes(), cfg_.score_thresh, d_res_.get(), s);

  // --- segmentation masks (argmax, open, close, component area/perimeter filter) -------------
  launch_seg_planes(engine_.lane(), engine_.lane_c(), static_cast<int>(hw), planes_a_.get(), s);
  launch_clean_morph(planes_a_.get(), planes_b_.get(), W, H, 3, cfg_.open_k, cfg_.close_k, s);
  launch_cc(planes_a_.get(), W, H, 3, cc_parent_.get(), cc_stats_.get(), true, s);
  launch_cc_filter(planes_a_.get(), cc_parent_.get(), cc_stats_.get(), planes_b_.get(), W, H, 3,
                   cfg_.min_area, cfg_.min_perimeter, s);
  // cleaned masks: planes_b_ = {white, yellow, stop line}

  launch_leading_edge(planes_b_.get() + 2 * hw, W, H, cfg_.subsample_step, cols_, d_res_.get(), s);

  // --- thinning of white + yellow, skeleton labelling / compaction --------------------------
  launch_thin_init(planes_b_.get(), geom_, thin0_.get(), s);
  launch_thin_iterations(thin0_.get(), thin1_.get(), geom_.Wt, geom_.Ht, thin_counters_.get(),
                         cfg_.thin_max_iter, s);
  launch_thin_corners(thin0_.get(), thin1_.get(), geom_.Wt, geom_.Ht, s);
  launch_cc(thin0_.get(), geom_.Wt, geom_.Ht, 2, skel_parent_.get(), skel_stats_.get(), false, s);
  launch_skel_compact(thin0_.get(), skel_parent_.get(), skel_stats_.get(), geom_, cfg_.skel_min_len,
                      d_res_.get(), s);

  VCUDA(cudaMemcpyAsync(h_res_.get(), d_res_.get(), sizeof(FrameResults), cudaMemcpyDeviceToHost, s));
  VCUDA(cudaGetLastError());
}

void GpuPipeline::capture() {
  bool capturing = false;
  try {
    VCUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    capturing = true;
    enqueue_all(stream_);
    VCUDA(cudaStreamEndCapture(stream_, &graph_));
    capturing = false;
    VCUDA(cudaGraphInstantiateWithFlags(&graph_exec_, graph_, 0));
    VCUDA(cudaGraphLaunch(graph_exec_, stream_));  // warm replay
    VCUDA(cudaStreamSynchronize(stream_));
  } catch (const std::exception& e) {
    if (capturing) {
      cudaGraph_t dead = nullptr;
      cudaStreamEndCapture(stream_, &dead);
      if (dead) cudaGraphDestroy(dead);
    }
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    graph_exec_ = nullptr;
    graph_ = nullptr;
    cudaGetLastError();  // clear sticky capture errors
    log_(std::string("[pipeline] CUDA graph capture failed, running eagerly: ") + e.what());
  }
}

void GpuPipeline::launch() {
  cudaEventRecord(ev_start_, stream_);
  if (graph_exec_)
    VCUDA(cudaGraphLaunch(graph_exec_, stream_));
  else
    enqueue_all(stream_);
  cudaEventRecord(ev_end_, stream_);
}

void GpuPipeline::wait() {
  VCUDA(cudaEventSynchronize(ev_end_));
  cudaEventElapsedTime(&gpu_ms_, ev_start_, ev_end_);
}

}  // namespace vision_cpp
