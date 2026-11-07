#pragma once
// POD types shared between CUDA kernels and host code. No CUDA/ROS includes here so that the
// pure-CPU parts of the stack (NMS, skeleton ordering, IPM) stay independently testable.
#include <cstdint>

namespace vision_cpp {

constexpr int kMaxCand = 2048;      // detection candidates above score_thresh kept on the GPU
constexpr int kMaxSkel = 8192;      // skeleton pixels kept per lane mask
constexpr int kMaxCols = 2048;      // max (subsampled) image width for the stop-line edge array
constexpr int kThinMaxIter = 64;    // Zhang-Suen iteration cap (same as the python version)

// Type-erased read-only view on a TensorRT output (fp16 or fp32).
struct TensorRef {
  const void* ptr;
  bool half;
};

struct Candidate {                  // 32 bytes
  float x1, y1, x2, y2;
  float score;
  int label;
  int flat;                         // anchor * C + class, used as deterministic tie-break
  int pad_;
};

struct SkelPix {                    // 8 bytes
  int label;                        // component root (raster index of its first pixel)
  int16_t v;                        // row in (sub-sampled) full-mask coordinates
  int16_t u;                        // column in (sub-sampled) mask coordinates
};

// Everything the CPU needs after the GPU part of a frame. Lives in device memory, is filled by the
// kernels and copied to pinned host memory with ONE memcpy. The 4 counters come first so that a
// single 16-byte memset resets them.
struct FrameResults {
  int cand_count;
  int skel_count[2];                // [0]=white lane, [1]=yellow lane
  int pad_;
  Candidate cands[kMaxCand];
  SkelPix skel[2][kMaxSkel];
  int16_t stop_v[kMaxCols];         // lowest stop-line row per (sub-sampled) column, -1 if none
};

}  // namespace vision_cpp
