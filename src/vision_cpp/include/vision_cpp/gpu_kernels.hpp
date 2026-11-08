#pragma once
// Host-side launchers for every GPU stage. All launches are asynchronous on the given stream and
// allocation-free, so the whole sequence can be captured into one CUDA graph.
#include <cuda_runtime.h>

#include <cstdint>

#include "vision_cpp/types.hpp"

namespace vision_cpp {

// ---- model_helpers.py: preprocess / postprocess_detections / clean_mask_np -----------------------

// uint8 HWC (BGR/RGB, 3 or 4 channels, arbitrary row stride) -> normalised NCHW RGB (fp16 or fp32).
// Bilinear resize (cv2.INTER_LINEAR semantics) is applied only if source size != (dw, dh).
void launch_preprocess(const uint8_t* src, int sw, int sh, int stride, int channels, bool bgr,
                       void* dst, bool dst_half, int dw, int dh, const float mean[3],
                       const float stdv[3], cudaStream_t s);

// score = sqrt(sigmoid(cls) * sigmoid(ctr)) > thr  ->  compacted candidate list in res->cands.
void launch_det_candidates(TensorRef cls, TensorRef ctr, TensorRef boxes, int n_anchors,
                           int n_classes, float thr, FrameResults* res, cudaStream_t s);

// argmax over classes, written as 3 binary planes (class 1, 2, 3) of size H*W each.
void launch_seg_planes(TensorRef logits, int n_classes, int hw, uint8_t* planes, cudaStream_t s);

// open(open_k) -> close(close_k) on `planes` binary planes. Needs a scratch buffer of equal size.
// The result is left in `a`.
void launch_clean_morph(uint8_t* a, uint8_t* b, int W, int H, int planes, int open_k, int close_k,
                        cudaStream_t s);

// 8-connected component labelling by lock-free union-find. parent/stats need planes*W*H entries.
// After the call, parent[plane*W*H + i] is the root of pixel i (valid for foreground pixels) and
// stats[root] = (area, perimeter). Perimeter = pixels removed by a 3x3 erosion (if `perimeter`).
void launch_cc(const uint8_t* mask, int W, int H, int planes, int* parent, int2* stats,
               bool perimeter, cudaStream_t s);

// out = mask && area(component) >= min_area && perimeter(component) >= min_perimeter
void launch_cc_filter(const uint8_t* mask, const int* parent, const int2* stats, uint8_t* out,
                      int W, int H, int planes, int min_area, int min_perimeter, cudaStream_t s);

// ---- vision.py: leading_edge_extraction ------------------------------------------------------------
void launch_leading_edge(const uint8_t* mask, int W, int H, int step, int cols, FrameResults* res,
                         cudaStream_t s);

// ---- model_output.py: overlay ---------------------------------------------------------------------
// Writes a tightly-packed BGR frame with translucent class colours (classes 1/2/3 = red/green/blue).
void launch_overlay(const uint8_t* src, int fw, int fh, int stride, int channels, bool bgr,
                    TensorRef logits, int n_classes, int lh, int lw, float alpha, uint8_t* out,
                    cudaStream_t s);

}  // namespace vision_cpp
