#include "vision_cpp/ipm.hpp"

#include <cmath>

namespace vision_cpp {

void Ipm::init(const IpmParams& p) {
  p_ = p;
  const int h = p.full_h;
  alpha_.resize(h);
  sin_a_.resize(h);
  cos_a_.resize(h);
  for (int v = 0; v < h; ++v) {
    const double a = std::atan((v - p.cy) / p.fy);
    alpha_[v] = a;
    sin_a_[v] = std::sin(a);
    cos_a_[v] = std::cos(a);
  }
  ready_ = true;
}

size_t Ipm::project(const int* v, const int* u, size_t n, double bot_pitch, float* out) const {
  const double beta = p_.cam_pitch + bot_pitch;
  const double sb = std::sin(beta), cb = std::cos(beta);
  const double inv_fx = 1.0 / p_.fx;
  const int step = p_.subsample_step;
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    const int vf = v[i] * step;
    if (vf < 0 || vf >= p_.full_h) continue;
    const double theta = alpha_[vf] + beta;
    if (theta < p_.min_angle) continue;  // `angles >= min_angle`

    const double sa = sin_a_[vf], ca = cos_a_[vf];
    const double st = sa * cb + ca * sb;  // sin(alpha + beta)
    const double ct = ca * cb - sa * sb;  // cos(alpha + beta)
    if (st == 0.0) continue;
    const double fwd = p_.cam_height * ct / st;  // cam_height / tan(theta)
    if (!(fwd > 0.0 && fwd <= p_.max_range)) continue;

    const double uf = static_cast<double>(u[i]) * step;
    out[3 * k + 0] = static_cast<float>(fwd + p_.cam_x);
    out[3 * k + 1] = static_cast<float>((p_.cx - uf) * inv_fx * fwd * ca / ct);
    out[3 * k + 2] = 0.f;
    ++k;
  }
  return k;
}

}  // namespace vision_cpp
