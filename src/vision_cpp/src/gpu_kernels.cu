#include <cuda_fp16.h>

#include <algorithm>

#include "vision_cpp/gpu_kernels.hpp"

namespace vision_cpp {
namespace {

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ldv(TensorRef t, size_t i) {
  return t.half ? __half2float(static_cast<const __half*>(t.ptr)[i])
                : static_cast<const float*>(t.ptr)[i];
}
__device__ __forceinline__ void stv(void* p, bool half, size_t i, float v) {
  if (half)
    static_cast<__half*>(p)[i] = __float2half(v);
  else
    static_cast<float*>(p)[i] = v;
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// =================================================================================================
// preprocess: uint8 HWC -> normalised NCHW RGB.   x*scale - bias == (x/255 - mean)/std
// =================================================================================================
__global__ void k_preprocess(const uint8_t* __restrict__ src, int sw, int sh, int stride, int sc,
                             int ri, int gi, int bi, void* dst, bool half, int dw, int dh,
                             float3 scale, float3 bias) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;

  float r, g, b;
  if (sw == dw && sh == dh) {
    const uint8_t* p = src + (size_t)y * stride + (size_t)x * sc;
    r = p[ri];
    g = p[gi];
    b = p[bi];
  } else {  // bilinear, half-pixel centres like cv2.INTER_LINEAR
    const float fx = (x + 0.5f) * sw / dw - 0.5f;
    const float fy = (y + 0.5f) * sh / dh - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    const float ax = fx - x0, ay = fy - y0;
    const int x1 = min(max(x0 + 1, 0), sw - 1), y1 = min(max(y0 + 1, 0), sh - 1);
    x0 = min(max(x0, 0), sw - 1);
    y0 = min(max(y0, 0), sh - 1);
    const uint8_t* p00 = src + (size_t)y0 * stride + (size_t)x0 * sc;
    const uint8_t* p01 = src + (size_t)y0 * stride + (size_t)x1 * sc;
    const uint8_t* p10 = src + (size_t)y1 * stride + (size_t)x0 * sc;
    const uint8_t* p11 = src + (size_t)y1 * stride + (size_t)x1 * sc;
    auto bil = [&](int c) {
      return (1.f - ay) * ((1.f - ax) * p00[c] + ax * p01[c]) + ay * ((1.f - ax) * p10[c] + ax * p11[c]);
    };
    r = bil(ri);
    g = bil(gi);
    b = bil(bi);
  }
  const size_t plane = (size_t)dw * dh, o = (size_t)y * dw + x;
  stv(dst, half, o, r * scale.x - bias.x);
  stv(dst, half, plane + o, g * scale.y - bias.y);
  stv(dst, half, 2 * plane + o, b * scale.z - bias.z);
}

// =================================================================================================
// detections: threshold + stream-compaction (N*C elements, trivially small)
// =================================================================================================
__global__ void k_det(TensorRef cls, TensorRef ctr, TensorRef boxes, int N, int C, float thr,
                      FrameResults* res) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= N) return;
  const float cv = sigmoidf(ldv(ctr, a));
  for (int c = 0; c < C; ++c) {
    const float p = sigmoidf(ldv(cls, (size_t)a * C + c));
    const float s = sqrtf(p * cv);
    if (s > thr) {
      const int k = atomicAdd(&res->cand_count, 1);
      if (k < kMaxCand) {
        Candidate& d = res->cands[k];
        d.x1 = ldv(boxes, (size_t)a * 4 + 0);
        d.y1 = ldv(boxes, (size_t)a * 4 + 1);
        d.x2 = ldv(boxes, (size_t)a * 4 + 2);
        d.y2 = ldv(boxes, (size_t)a * 4 + 3);
        d.score = s;
        d.label = c;
        d.flat = a * C + c;
      }
    }
  }
}

// =================================================================================================
// segmentation: argmax -> 3 binary planes
// =================================================================================================
__global__ void k_seg_planes(TensorRef logits, int C, int HW, uint8_t* planes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= HW) return;
  int best = 0;
  float bv = ldv(logits, i);
  for (int c = 1; c < C; ++c) {
    const float v = ldv(logits, (size_t)c * HW + i);
    if (v > bv) {
      bv = v;
      best = c;
    }
  }
  planes[i] = (best == 1);
  planes[(size_t)HW + i] = (best == 2);
  planes[2 * (size_t)HW + i] = (best == 3);
}

// =================================================================================================
// morphology (rectangular SE, separable). Out-of-image pixels are neutral, like cv2's default
// border value for erode (+inf) / dilate (-inf).
// =================================================================================================
template <bool ERODE>
__global__ void k_morph_h(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H,
                          int r) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t base = (size_t)blockIdx.z * W * H + (size_t)y * W;
  uint8_t acc = ERODE ? 1 : 0;
  for (int dx = -r; dx <= r; ++dx) {
    const int xx = x + dx;
    if (xx < 0 || xx >= W) continue;
    const uint8_t v = in[base + xx];
    acc = ERODE ? (acc & v) : (acc | v);
  }
  out[base + x] = acc;
}

template <bool ERODE>
__global__ void k_morph_v(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H,
                          int r) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t base = (size_t)blockIdx.z * W * H;
  uint8_t acc = ERODE ? 1 : 0;
  for (int dy = -r; dy <= r; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= H) continue;
    const uint8_t v = in[base + (size_t)yy * W + x];
    acc = ERODE ? (acc & v) : (acc | v);
  }
  out[base + (size_t)y * W + x] = acc;
}

// =================================================================================================
// connected components: lock-free union-find (root = smallest raster index of the component)
// All indices are LOCAL to a plane; plane offset = blockIdx.z * W * H.
// =================================================================================================
__device__ __forceinline__ int uf_find(int* p, int x) {
  for (;;) {
    const int px = __ldcg(&p[x]);
    if (px == x) return x;
    const int ppx = __ldcg(&p[px]);
    if (ppx != px) p[x] = ppx;  // path halving; x is not a root so no CAS can target it
    x = ppx;
  }
}

__device__ __forceinline__ void uf_union(int* p, int a, int b) {
  for (;;) {
    a = uf_find(p, a);
    b = uf_find(p, b);
    if (a == b) return;
    if (a < b) {
      const int t = a;
      a = b;
      b = t;
    }  // a = larger root -> hang it under the smaller one
    if (atomicCAS(&p[a], a, b) == a) return;
  }
}

__global__ void k_cc_init(const uint8_t* __restrict__ mask, int* parent, int2* stats, int W, int H) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t ps = (size_t)W * H, i = (size_t)y * W + x;
  if (mask[blockIdx.z * ps + i]) {  // only foreground is ever touched later
    parent[blockIdx.z * ps + i] = (int)i;
    stats[blockIdx.z * ps + i] = make_int2(0, 0);
  }
}

__global__ void k_cc_merge(const uint8_t* __restrict__ mask, int* parent, int W, int H) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t ps = (size_t)W * H;
  const uint8_t* m = mask + blockIdx.z * ps;
  int* p = parent + blockIdx.z * ps;
  const int i = y * W + x;
  if (!m[i]) return;
  if (x > 0 && m[i - 1]) uf_union(p, i, i - 1);
  if (y > 0) {
    if (x > 0 && m[i - W - 1]) uf_union(p, i, i - W - 1);
    if (m[i - W]) uf_union(p, i, i - W);
    if (x < W - 1 && m[i - W + 1]) uf_union(p, i, i - W + 1);
  }
}

__global__ void k_cc_stats(const uint8_t* __restrict__ mask, int* parent, int2* stats, int W, int H,
                           bool perimeter) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t ps = (size_t)W * H;
  const uint8_t* m = mask + blockIdx.z * ps;
  int* p = parent + blockIdx.z * ps;
  int* st = reinterpret_cast<int*>(stats + blockIdx.z * ps);
  const int i = y * W + x;
  if (!m[i]) return;
  const int r = uf_find(p, i);
  p[i] = r;  // flatten for the filter / compaction kernels
  atomicAdd(&st[2 * r], 1);
  if (perimeter) {
    bool edge = false;  // removed by a 3x3 erosion with neutral borders
    for (int dy = -1; dy <= 1 && !edge; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        if (!m[yy * W + xx]) {
          edge = true;
          break;
        }
      }
    if (edge) atomicAdd(&st[2 * r + 1], 1);
  }
}

__global__ void k_cc_filter(const uint8_t* __restrict__ mask, const int* __restrict__ parent,
                            const int2* __restrict__ stats, uint8_t* out, int W, int H, int min_area,
                            int min_perim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const size_t ps = (size_t)W * H, i = blockIdx.z * ps + (size_t)y * W + x;
  uint8_t o = 0;
  if (mask[i]) {
    const int2 s = stats[blockIdx.z * ps + parent[i]];
    o = (s.x >= min_area && s.y >= min_perim);
  }
  out[i] = o;
}

// =================================================================================================
// stop line: lowest foreground row for each (sub-sampled) column
// =================================================================================================
__global__ void k_leading_edge(const uint8_t* __restrict__ mask, int W, int H, int step, int cols,
                               FrameResults* res) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= cols) return;
  const int x = u * step, Hs = (H + step - 1) / step;
  int16_t r = -1;
  for (int v = Hs - 1; v >= 0; --v)
    if (mask[(size_t)(v * step) * W + x]) {
      r = (int16_t)v;
      break;
    }
  res->stop_v[u] = r;
}

// =================================================================================================
// overlay (model_output.py)
// =================================================================================================
__global__ void k_overlay(const uint8_t* __restrict__ src, int fw, int fh, int stride, int sc,
                          int ri, int gi, int bi, TensorRef logits, int C, int lh, int lw,
                          float alpha, uint8_t* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= fw || y >= fh) return;
  const uint8_t* p = src + (size_t)y * stride + (size_t)x * sc;
  float B = p[bi], G = p[gi], R = p[ri];

  const int lx = min((int)((long long)x * lw / fw), lw - 1);  // cv2.INTER_NEAREST
  const int ly = min((int)((long long)y * lh / fh), lh - 1);
  const size_t idx = (size_t)ly * lw + lx, hw = (size_t)lh * lw;
  int best = 0;
  float bv = ldv(logits, idx);
  for (int c = 1; c < C; ++c) {
    const float v = ldv(logits, c * hw + idx);
    if (v > bv) {
      bv = v;
      best = c;
    }
  }
  if (best > 0) {
    const float cb = best == 3 ? 255.f : 0.f, cg = best == 2 ? 255.f : 0.f,
                cr = best == 1 ? 255.f : 0.f;  // classes > 3: black, like the python version
    B = B * (1.f - alpha) + cb * alpha;
    G = G * (1.f - alpha) + cg * alpha;
    R = R * (1.f - alpha) + cr * alpha;
  }
  uint8_t* o = out + ((size_t)y * fw + x) * 3;
  o[0] = (uint8_t)fminf(255.f, rintf(B));
  o[1] = (uint8_t)fminf(255.f, rintf(G));
  o[2] = (uint8_t)fminf(255.f, rintf(R));
}

}  // namespace

// =================================================================================================
// launchers
// =================================================================================================
void launch_preprocess(const uint8_t* src, int sw, int sh, int stride, int channels, bool bgr,
                       void* dst, bool dst_half, int dw, int dh, const float mean[3],
                       const float stdv[3], cudaStream_t s) {
  const int ri = bgr ? 2 : 0, gi = 1, bi = bgr ? 0 : 2;
  const float3 scale = make_float3(1.f / (255.f * stdv[0]), 1.f / (255.f * stdv[1]),
                                   1.f / (255.f * stdv[2]));
  const float3 bias = make_float3(mean[0] / stdv[0], mean[1] / stdv[1], mean[2] / stdv[2]);
  const dim3 b(32, 8), g(cdiv(dw, 32), cdiv(dh, 8));
  k_preprocess<<<g, b, 0, s>>>(src, sw, sh, stride, channels, ri, gi, bi, dst, dst_half, dw, dh,
                               scale, bias);
}

void launch_det_candidates(TensorRef cls, TensorRef ctr, TensorRef boxes, int n_anchors,
                           int n_classes, float thr, FrameResults* res, cudaStream_t s) {
  k_det<<<cdiv(n_anchors, 128), 128, 0, s>>>(cls, ctr, boxes, n_anchors, n_classes, thr, res);
}

void launch_seg_planes(TensorRef logits, int n_classes, int hw, uint8_t* planes, cudaStream_t s) {
  k_seg_planes<<<cdiv(hw, 256), 256, 0, s>>>(logits, n_classes, hw, planes);
}

void launch_clean_morph(uint8_t* a, uint8_t* b, int W, int H, int planes, int open_k, int close_k,
                        cudaStream_t s) {
  const dim3 bl(32, 8), g(cdiv(W, 32), cdiv(H, 8), planes);
  auto pass = [&](bool erode, int k) {
    const int r = k / 2;
    if (erode) {
      k_morph_h<true><<<g, bl, 0, s>>>(a, b, W, H, r);
      k_morph_v<true><<<g, bl, 0, s>>>(b, a, W, H, r);
    } else {
      k_morph_h<false><<<g, bl, 0, s>>>(a, b, W, H, r);
      k_morph_v<false><<<g, bl, 0, s>>>(b, a, W, H, r);
    }
  };
  if (open_k > 1) {
    pass(true, open_k);
    pass(false, open_k);
  }
  if (close_k > 1) {
    pass(false, close_k);
    pass(true, close_k);
  }
}

void launch_cc(const uint8_t* mask, int W, int H, int planes, int* parent, int2* stats,
               bool perimeter, cudaStream_t s) {
  const dim3 bl(32, 8), g(cdiv(W, 32), cdiv(H, 8), planes);
  k_cc_init<<<g, bl, 0, s>>>(mask, parent, stats, W, H);
  k_cc_merge<<<g, bl, 0, s>>>(mask, parent, W, H);
  k_cc_stats<<<g, bl, 0, s>>>(mask, parent, stats, W, H, perimeter);
}

void launch_cc_filter(const uint8_t* mask, const int* parent, const int2* stats, uint8_t* out,
                      int W, int H, int planes, int min_area, int min_perimeter, cudaStream_t s) {
  const dim3 bl(32, 8), g(cdiv(W, 32), cdiv(H, 8), planes);
  k_cc_filter<<<g, bl, 0, s>>>(mask, parent, stats, out, W, H, min_area, min_perimeter);
}

void launch_leading_edge(const uint8_t* mask, int W, int H, int step, int cols, FrameResults* res,
                         cudaStream_t s) {
  k_leading_edge<<<cdiv(cols, 128), 128, 0, s>>>(mask, W, H, step, cols, res);
}

void launch_overlay(const uint8_t* src, int fw, int fh, int stride, int channels, bool bgr,
                    TensorRef logits, int n_classes, int lh, int lw, float alpha, uint8_t* out,
                    cudaStream_t s) {
  const int ri = bgr ? 2 : 0, gi = 1, bi = bgr ? 0 : 2;
  const dim3 b(32, 8), g(cdiv(fw, 32), cdiv(fh, 8));
  k_overlay<<<g, b, 0, s>>>(src, fw, fh, stride, channels, ri, gi, bi, logits, n_classes, lh, lw,
                            alpha, out);
}

}  // namespace vision_cpp
