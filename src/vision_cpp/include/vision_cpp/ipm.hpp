#pragma once
#include <cstddef>
#include <vector>

namespace vision_cpp {

struct IpmParams {
  double cx = 0, cy = 0, fx = 1, fy = 1;
  double cam_pitch = 0.4, cam_height = 1.606;
  double min_angle = 0.01, max_range = 8.0, cam_x = 0.02;
  int subsample_step = 1;
  int full_h = 544;  // number of full-resolution image rows
};

// Port of ipm.py::project_pixels_np.
//
// Per-row quantities (alpha = atan((v-cy)/fy), sin/cos alpha) do not depend on the frame, so they
// are tabulated once when the intrinsics arrive. Per frame only sin/cos of the (cam+bot) pitch are
// evaluated, and sin/cos(alpha+beta) come from the angle-addition identities: the per-point cost is
// a handful of multiplies and one division, no transcendental calls.
class Ipm {
 public:
  void init(const IpmParams& p);
  bool ready() const { return ready_; }

  // v,u: pixel coordinates in the sub-sampled mask. Writes float32 (x,y,z) triples to `out`
  // (capacity >= 3*n floats) and returns the number of points kept.
  size_t project(const int* v, const int* u, size_t n, double bot_pitch, float* out) const;

 private:
  IpmParams p_;
  std::vector<double> alpha_, sin_a_, cos_a_;
  bool ready_ = false;
};

}  // namespace vision_cpp
