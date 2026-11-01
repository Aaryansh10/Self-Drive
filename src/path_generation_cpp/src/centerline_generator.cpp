#include "path_generation_cpp/centerline_generator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace centerline {

namespace {

using Vec = std::vector<double>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline std::size_t idx(Side s) { return static_cast<std::size_t>(s); }
inline double sq(double v) { return v * v; }

// Candidate offsets per line type: (lane, fraction of road width, offset toward right?, kind)
const std::array<OffsetSpec, 2> kWhiteLeft = {{
    {Side::Left, 0.25, true, Kind::Near},
    {Side::Right, 0.75, true, Kind::Far},
}};
const std::array<OffsetSpec, 2> kWhiteRight = {{
    {Side::Right, 0.25, false, Kind::Near},
    {Side::Left, 0.75, false, Kind::Far},
}};
const std::array<OffsetSpec, 2> kYellow = {{
    {Side::Left, 0.25, false, Kind::Yellow},
    {Side::Right, 0.25, true, Kind::Yellow},
}};

Vec linspace(double a, double b, int n) {
  Vec out(static_cast<std::size_t>(std::max(n, 0)));
  if (n <= 0) return out;
  if (n == 1) {
    out[0] = a;
    return out;
  }
  const double step = (b - a) / (n - 1);
  for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = a + step * i;
  out.back() = b;
  return out;
}

// torch.searchsorted(left) + clamp(1, N-1) linear interpolation. xp must be non-decreasing, N >= 2.
double interp1(const Vec& xp, const Vec& fp, double xq) {
  const int n = static_cast<int>(xp.size());
  int i = static_cast<int>(std::lower_bound(xp.begin(), xp.end(), xq) - xp.begin());
  i = std::clamp(i, 1, n - 1);
  const double xl = xp[i - 1];
  const double xr = xp[i];
  const double ratio = (xq - xl) / std::max(xr - xl, 1e-6);
  return fp[i - 1] + ratio * (fp[i] - fp[i - 1]);
}

Vec interpolate(const Vec& xq, const Vec& xp, const Vec& fp) {
  Vec out(xq.size());
  for (std::size_t i = 0; i < xq.size(); ++i) out[i] = interp1(xp, fp, xq[i]);
  return out;
}

// Least-squares pseudo-inverse of the Vandermonde matrix A (rows: sample offsets, cols: powers
// 0..p-1), i.e. (A^T A)^-1 A^T, returned as p x w. Equivalent to np.linalg.pinv for full column rank.
std::vector<Vec> vanderPinv(const Vec& offsets, int p) {
  const int w = static_cast<int>(offsets.size());
  std::vector<Vec> A(static_cast<std::size_t>(w), Vec(static_cast<std::size_t>(p)));
  for (int i = 0; i < w; ++i) {
    double v = 1.0;
    for (int k = 0; k < p; ++k) {
      A[i][k] = v;
      v *= offsets[i];
    }
  }

  // Augmented [A^T A | I], Gauss-Jordan with partial pivoting.
  std::vector<Vec> M(static_cast<std::size_t>(p), Vec(static_cast<std::size_t>(2 * p), 0.0));
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int k = 0; k < w; ++k) s += A[k][i] * A[k][j];
      M[i][j] = s;
    }
    M[i][p + i] = 1.0;
  }
  for (int c = 0; c < p; ++c) {
    int piv = c;
    for (int r = c + 1; r < p; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
    std::swap(M[c], M[piv]);
    const double d = M[c][c];
    for (int j = 0; j < 2 * p; ++j) M[c][j] /= d;
    for (int r = 0; r < p; ++r) {
      if (r == c) continue;
      const double f = M[r][c];
      if (f == 0.0) continue;
      for (int j = 0; j < 2 * p; ++j) M[r][j] -= f * M[c][j];
    }
  }

  std::vector<Vec> pinv(static_cast<std::size_t>(p), Vec(static_cast<std::size_t>(w), 0.0));
  for (int i = 0; i < p; ++i)
    for (int k = 0; k < w; ++k) {
      double s = 0.0;
      for (int j = 0; j < p; ++j) s += M[i][p + j] * A[k][j];
      pinv[i][k] = s;
    }
  return pinv;
}

double factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Construction / memory
// ---------------------------------------------------------------------------------------------

CenterlineGenerator::CenterlineGenerator(const Params& params) : p_(params) {
  p_.width_stale_decay_frames = std::max(1, p_.width_stale_decay_frames);
  use_normal_blend_ = p_.normal_blend_tx_min.has_value() && p_.normal_blend_tx_max.has_value();

  max_bins_ = static_cast<int>(std::ceil(p_.max_range / p_.bin_width));
  min_offset_ = p_.nominal_road_width * p_.min_offset_fraction;
  num_out_ = static_cast<int>(std::ceil((p_.max_range - p_.path_start) / p_.path_spacing)) + 1;
  d_grid_ = linspace(p_.path_start, p_.max_range, num_out_);

  sg_polyorder_ = std::max(1, p_.sg_polyorder);
  sg_min_window_ = sg_polyorder_ + 1;
  if (sg_min_window_ % 2 == 0) ++sg_min_window_;

  resetMemory();

  // Pre-build the SG kernels for full-length lines so the first frame doesn't pay for them.
  const int wp = oddWindow(p_.position_window_m, 100000);
  const int wt = oddWindow(p_.tangent_window_m, 100000);
  if (wt >= sg_min_window_) {
    sgKernel(std::max(wp, sg_min_window_), 0);
    sgKernel(wt, 1);
  }
}

void CenterlineGenerator::resetMemory() {
  const auto nb = static_cast<std::size_t>(max_bins_);
  const auto no = static_cast<std::size_t>(num_out_);
  prev_left_y_.assign(nb, kNaN);
  prev_right_y_.assign(nb, kNaN);
  road_width_.assign(nb, p_.nominal_road_width);
  width_stale_.assign(nb, 0.0);
  for (std::size_t s = 0; s < 2; ++s) {
    track_x_[s].assign(no, kNaN);
    track_y_[s].assign(no, kNaN);
  }
  bad_frame_streak_ = 0;
}

void CenterlineGenerator::failedFrame() {
  ++bad_frame_streak_;
  if (bad_frame_streak_ >= p_.max_bad_frame_streak) resetMemory();
}

void CenterlineGenerator::updateMemory(const Vec& y, const std::vector<int>& bins,
                                       const std::vector<char>& pt_is_left) {
  const auto nb = static_cast<std::size_t>(max_bins_);
  Vec lsum(nb, 0.0), rsum(nb, 0.0);
  std::vector<int> lcnt(nb, 0), rcnt(nb, 0);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const auto b = static_cast<std::size_t>(bins[i]);
    if (pt_is_left[i]) {
      lsum[b] += y[i];
      ++lcnt[b];
    } else {
      rsum[b] += y[i];
      ++rcnt[b];
    }
  }

  const double a = p_.track_ema_alpha;
  const double aw = p_.width_ema_alpha;
  const int decay_frames = p_.width_stale_decay_frames;

  std::vector<char> trusted(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const bool l_has = lcnt[b] >= p_.min_bin_points;
    const bool r_has = rcnt[b] >= p_.min_bin_points;
    const double lmean = lsum[b] / std::max(lcnt[b], 1);
    const double rmean = rsum[b] / std::max(rcnt[b], 1);

    if (l_has) prev_left_y_[b] = std::isnan(prev_left_y_[b]) ? lmean : a * lmean + (1.0 - a) * prev_left_y_[b];
    if (r_has) prev_right_y_[b] = std::isnan(prev_right_y_[b]) ? rmean : a * rmean + (1.0 - a) * prev_right_y_[b];

    const double raw = lmean - rmean;
    const bool plausible = raw >= p_.min_road_width && raw <= p_.max_road_width;
    const double bin_x = (static_cast<double>(b) + 0.5) * p_.bin_width;
    trusted[b] = bin_x <= p_.width_update_max_x;
    const bool confirmed = l_has && r_has && plausible && trusted[b];

    double& rw = road_width_[b];
    const double target =
        aw * std::clamp(raw, p_.min_road_width, p_.max_road_width) + (1.0 - aw) * rw;
    const double step = std::clamp(target - rw, -p_.max_width_update, p_.max_width_update);
    if (confirmed) rw += step;

    width_stale_[b] = confirmed ? 0.0 : width_stale_[b] + 1.0;
    if (width_stale_[b] > decay_frames) rw += p_.width_stale_decay_alpha * (p_.nominal_road_width - rw);
  }

  // Bins beyond the trusted range inherit the mean width of the fresh, trusted bins.
  double fresh_sum = 0.0;
  int fresh_n = 0;
  for (std::size_t b = 0; b < nb; ++b) {
    if (trusted[b] && width_stale_[b] <= decay_frames) {
      fresh_sum += road_width_[b];
      ++fresh_n;
    }
  }
  if (fresh_n > 0) {
    const double mean = fresh_sum / fresh_n;
    for (std::size_t b = 0; b < nb; ++b)
      if (!trusted[b]) road_width_[b] = mean;
  }
}

// ---------------------------------------------------------------------------------------------
// Line splitting / labelling
// ---------------------------------------------------------------------------------------------

CenterlineGenerator::LineSplit CenterlineGenerator::splitLines(const Vec& x, const Vec& y) const {
  LineSplit ls;
  ls.line_id.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (i == 0 || std::hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > p_.max_gap) {
      ls.start.push_back(static_cast<int>(i));
      ls.count.push_back(0);
    }
    ++ls.count.back();
    ls.line_id[i] = static_cast<int>(ls.start.size()) - 1;
  }
  return ls;
}

int CenterlineGenerator::distBin(double x) const {
  const double q = x / p_.bin_width;
  if (!(q > 0.0)) return 0;  // also catches NaN
  if (q >= max_bins_) return max_bins_ - 1;
  return static_cast<int>(q);
}

std::vector<char> CenterlineGenerator::labelLines(const Vec& y, const LineSplit& ls,
                                                  const std::vector<int>& bins) const {
  std::vector<int> left_votes(ls.start.size(), 0);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const auto b = static_cast<std::size_t>(bins[i]);
    double pl = prev_left_y_[b];
    double pr = prev_right_y_[b];
    const double rw = road_width_[b];
    const bool unk_l = std::isnan(pl);
    const bool unk_r = std::isnan(pr);

    bool vote_left;
    if (unk_l && unk_r) {
      vote_left = y[i] > 0.0;
    } else {
      if (unk_l) pl = pr + rw;
      if (unk_r) pr = pl - rw;
      vote_left = std::fabs(y[i] - pl) <= std::fabs(y[i] - pr);
    }
    if (vote_left) ++left_votes[static_cast<std::size_t>(ls.line_id[i])];
  }

  std::vector<char> is_left(ls.start.size());
  for (std::size_t l = 0; l < ls.start.size(); ++l) is_left[l] = left_votes[l] * 2 >= ls.count[l];
  return is_left;
}

// ---------------------------------------------------------------------------------------------
// Candidate generation
// ---------------------------------------------------------------------------------------------

int CenterlineGenerator::oddWindow(double meters, int n) const {
  int w = static_cast<int>(std::nearbyint(meters / p_.resample_step));  // round-half-even, like Python
  w = std::min(w, n);
  return (w % 2 == 1) ? w : w - 1;
}

bool CenterlineGenerator::resampleLine(const Vec& x, const Vec& y, Vec& rx, Vec& ry) const {
  const std::size_t n = x.size();
  Vec arclen(n, 0.0);
  for (std::size_t i = 1; i < n; ++i)
    arclen[i] = arclen[i - 1] + std::hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  const double total = arclen.back();
  if (total < 3.0 * p_.resample_step) return false;
  const int m = static_cast<int>(total / p_.resample_step) + 1;
  const Vec query = linspace(0.0, total, m);
  rx = interpolate(query, arclen, x);
  ry = interpolate(query, arclen, y);
  return true;
}

void CenterlineGenerator::inwardNormal(double tx, double ty, bool toward_right, double& nx,
                                       double& ny) const {
  double gx, gy, lat_y;
  if (toward_right) {
    gx = ty;
    gy = -tx;
    lat_y = -1.0;
  } else {
    gx = -ty;
    gy = tx;
    lat_y = 1.0;
  }
  if (!use_normal_blend_) {
    nx = gx;
    ny = gy;
    return;
  }
  const double lo = *p_.normal_blend_tx_min;
  const double span = std::max(*p_.normal_blend_tx_max - lo, 1e-6);
  const double w_geo = std::clamp((tx - lo) / span, 0.0, 1.0);
  const double bx = w_geo * gx;
  const double by = w_geo * gy + (1.0 - w_geo) * lat_y;
  const double nn = std::max(std::hypot(bx, by), 1e-6);
  nx = bx / nn;
  ny = by / nn;
}

void CenterlineGenerator::lineCenterline(Vec x, Vec y, const std::array<OffsetSpec, 2>& offsets,
                                         std::vector<Candidate>& out) {
  if (x.size() < 3) return;

  if (std::hypot(x.front(), y.front()) > std::hypot(x.back(), y.back())) {
    std::reverse(x.begin(), x.end());
    std::reverse(y.begin(), y.end());
  }

  Vec rx, ry;
  if (!resampleLine(x, y, rx, ry)) return;
  const int n = static_cast<int>(rx.size());

  // Position uses a short window; the tangent uses a longer one, because any tangent error is
  // multiplied by the offset when the path is pushed sideways.
  int wp = oddWindow(p_.position_window_m, n);
  const int wt = oddWindow(p_.tangent_window_m, n);

  Vec xs, ys, tx, ty;
  if (wt >= sg_min_window_) {
    wp = std::max(wp, sg_min_window_);
    xs = sgApply(rx, wp, 0);
    ys = sgApply(ry, wp, 0);
    tx = sgApply(rx, wt, 1);
    ty = sgApply(ry, wt, 1);
  } else {
    int w = std::min(7, n);
    if (w % 2 == 0) --w;
    xs = boxSmooth(rx, w);
    ys = boxSmooth(ry, w);
    tx = gradient(xs);
    ty = gradient(ys);
  }

  const auto un = static_cast<std::size_t>(n);
  Vec road_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    const double norm = std::hypot(tx[i], ty[i]) + 1e-5;
    tx[i] /= norm;
    ty[i] /= norm;
    road_w[i] = road_width_[static_cast<std::size_t>(distBin(xs[i]))];
  }

  std::vector<std::size_t> order(un);
  Vec radial(un);
  for (const OffsetSpec& spec : offsets) {
    Vec cx(un), cy(un);
    for (std::size_t i = 0; i < un; ++i) {
      double nx, ny;
      inwardNormal(tx[i], ty[i], spec.toward_right, nx, ny);
      const double offset = std::max(road_w[i] * spec.fraction, min_offset_ * spec.fraction);
      cx[i] = xs[i] + nx * offset;
      cy[i] = ys[i] + ny * offset;
      radial[i] = std::hypot(cx[i], cy[i]);
    }
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return radial[a] < radial[b]; });

    Candidate c;
    c.lane = spec.lane;
    c.kind = spec.kind;
    c.fraction = spec.fraction;
    c.x.resize(un);
    c.y.resize(un);
    c.radial.resize(un);
    for (std::size_t i = 0; i < un; ++i) {
      c.x[i] = cx[order[i]];
      c.y[i] = cy[order[i]];
      // Tiny ramp keeps the radial axis strictly increasing so interpolation can't divide by ~0.
      c.radial[i] = radial[order[i]] + static_cast<double>(i) * 1e-4;
    }
    out.push_back(std::move(c));
  }
}

std::vector<char> CenterlineGenerator::yellowGate(const Vec& x, const Vec& y) const {
  std::vector<char> keep(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto b = static_cast<std::size_t>(distBin(x[i]));
    const double l = prev_left_y_[b];
    const double r = prev_right_y_[b];
    const double width = road_width_[b];
    const bool kl = !std::isnan(l);
    const bool kr = !std::isnan(r);

    double mid = kNaN;
    if (kl && kr) mid = 0.5 * (l + r);
    else if (kl) mid = l - 0.5 * width;
    else if (kr) mid = r + 0.5 * width;

    keep[i] = std::isnan(mid) || std::fabs(y[i] - mid) <= p_.yellow_gate_fraction * width;
  }
  return keep;
}

void CenterlineGenerator::yellowCandidates(const std::vector<Point2>& yellow,
                                           std::vector<Candidate>& out) {
  Vec x(yellow.size()), y(yellow.size());
  for (std::size_t i = 0; i < yellow.size(); ++i) {
    x[i] = yellow[i].x;
    y[i] = yellow[i].y;
  }
  const std::vector<char> keep = yellowGate(x, y);

  Vec gx, gy;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (keep[i]) {
      gx.push_back(x[i]);
      gy.push_back(y[i]);
    }
  }
  if (gx.size() < 3) return;

  const LineSplit ls = splitLines(gx, gy);
  for (std::size_t l = 0; l < ls.start.size(); ++l) {
    const auto s = static_cast<std::ptrdiff_t>(ls.start[l]);
    const auto c = static_cast<std::ptrdiff_t>(ls.count[l]);
    lineCenterline(Vec(gx.begin() + s, gx.begin() + s + c), Vec(gy.begin() + s, gy.begin() + s + c),
                   kYellow, out);
  }
}

// ---------------------------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------------------------

CenterlineGenerator::Vec CenterlineGenerator::maskedSmooth(const Vec& values,
                                                           const std::vector<char>& valid,
                                                           int window) const {
  // Windowed local-linear smoother that ignores invalid samples (see the Python original).
  const int n = static_cast<int>(values.size());
  int nvalid = 0;
  for (char v : valid) nvalid += v ? 1 : 0;

  int w = std::min(window, nvalid);
  if (w % 2 == 0) --w;
  if (w < 3) return values;
  const int h = w / 2;

  Vec out(values.size());
  for (int i = 0; i < n; ++i) {
    double s0 = 0, s1 = 0, s2 = 0, sv = 0, stv = 0;
    for (int t = -h; t <= h; ++t) {
      const int j = i + t;
      if (j < 0 || j >= n || !valid[static_cast<std::size_t>(j)]) continue;
      const double v = values[static_cast<std::size_t>(j)];
      s0 += 1.0;
      s1 += t;
      s2 += static_cast<double>(t) * t;
      sv += v;
      stv += t * v;
    }
    const double det = s0 * s2 - s1 * s1;
    const double linear = (s2 * sv - s1 * stv) / std::max(det, 1e-6);
    const double box = sv / std::max(s0, 1e-6);
    out[static_cast<std::size_t>(i)] = det > 0.5 ? linear : box;
  }
  return out;
}

void CenterlineGenerator::lockNearField(Vec& x, Vec& y) {
  // Re-projects the first two points onto the heading of points 2..4 to prevent hooks at turns.
  if (x.size() < 5) return;

  double hx = x[4] - x[2];
  double hy = y[4] - y[2];
  const double hn = std::hypot(hx, hy) + 1e-5;
  hx /= hn;
  hy /= hn;

  const double d0 = std::hypot(x[2] - x[0], y[2] - y[0]);
  const double d1 = std::hypot(x[2] - x[1], y[2] - y[1]);

  x[0] = x[2] - hx * d0;
  y[0] = y[2] - hy * d0;
  x[1] = x[2] - hx * d1;
  y[1] = y[2] - hy * d1;
}

Path CenterlineGenerator::fuseSide(Side side, const std::vector<const Candidate*>& candidates) {
  if (candidates.empty()) return {};

  const auto no = static_cast<std::size_t>(num_out_);
  Vec sum_x(no, 0.0), sum_y(no, 0.0), sum_w(no, 0.0);

  for (const Candidate* c : candidates) {
    const double lo = c->radial.front();
    const double hi = c->radial.back();

    // Reliability weight: a candidate offset by `fraction` of the road width carries
    // fraction * width_error lateral error, so weight by 1 / offset^2 (near/yellow = 1, far = 1/9).
    const double w = sq(ref_fraction_ / std::max(c->fraction, 1e-3));

    for (std::size_t g = 0; g < no; ++g) {
      const double d = d_grid_[g];
      if (d < lo || d > hi) continue;
      // Ramp the weight in/out over the candidate's radial span so a candidate appearing or
      // disappearing doesn't make the fused path jump.
      const double edge = std::min(d - lo, hi - d);
      const double ramp = std::clamp(edge / taper_len_, 0.0, 1.0);
      const double weight = w * (0.02 + 0.98 * ramp);
      sum_x[g] += weight * interp1(c->radial, c->x, d);
      sum_y[g] += weight * interp1(c->radial, c->y, d);
      sum_w[g] += weight;
    }
  }

  std::vector<char> valid(no);
  std::size_t n_valid = 0;
  Vec x_mean(no), y_mean(no);
  for (std::size_t g = 0; g < no; ++g) {
    valid[g] = sum_w[g] > 0.0;
    n_valid += valid[g] ? 1 : 0;
    const double den = std::max(sum_w[g], 1e-9);
    x_mean[g] = sum_x[g] / den;
    y_mean[g] = sum_y[g] / den;
  }
  if (n_valid < 3) return {};

  // Smooth on the FULL uniform grid with a validity mask, then compact.
  const Vec sx_full = maskedSmooth(x_mean, valid, p_.final_smooth_window);
  const Vec sy_full = maskedSmooth(y_mean, valid, p_.final_smooth_window);
  Vec sx, sy;
  sx.reserve(n_valid);
  sy.reserve(n_valid);
  std::vector<std::size_t> grid_idx;
  grid_idx.reserve(n_valid);
  for (std::size_t g = 0; g < no; ++g) {
    if (!valid[g]) continue;
    sx.push_back(sx_full[g]);
    sy.push_back(sy_full[g]);
    grid_idx.push_back(g);
  }

  // The path has to start close to the robot in forward distance.
  bool has_close = false;
  for (const Candidate* c : candidates)
    if (c->fraction <= ref_fraction_ + 1e-6) has_close = true;
  const double start_limit = has_close ? p_.near_field_range : p_.far_near_field_range;
  const double min_x = *std::min_element(sx.begin(), sx.end());
  if (static_cast<int>(sx.size()) < p_.min_near_pts || min_x > start_limit) return {};

  // EMA against the tracked path from previous frames.
  Vec& trk_x = track_x_[idx(side)];
  Vec& trk_y = track_y_[idx(side)];
  const double alpha = p_.centerline_ema_alpha;
  for (std::size_t j = 0; j < sx.size(); ++j) {
    const std::size_t g = grid_idx[j];
    if (!std::isnan(trk_x[g])) {
      sx[j] = alpha * sx[j] + (1.0 - alpha) * trk_x[g];
      sy[j] = alpha * sy[j] + (1.0 - alpha) * trk_y[g];
    }
  }

  lockNearField(sx, sy);

  Path path(sx.size());
  for (std::size_t j = 0; j < sx.size(); ++j) {
    trk_x[grid_idx[j]] = sx[j];
    trk_y[grid_idx[j]] = sy[j];
    path[j] = {sx[j], sy[j]};
  }
  return path;
}

Path CenterlineGenerator::anchorToOrigin(const Path& path) const {
  if (path.empty()) return path;
  if (std::hypot(path[0].x, path[0].y) < 1e-3) return path;
  return bezierLaneChange(path);
}

Path CenterlineGenerator::bezierLaneChange(const Path& path) const {
  // Replaces the start of the path with a cubic Bezier that leaves the origin along +x and joins
  // the lane centerline tangentially. Returns a path starting at (0, 0).
  if (path.size() < 3) return path;

  const int n = static_cast<int>(path.size());
  double max_r = 0.0;
  Vec r(path.size());
  for (std::size_t i = 0; i < path.size(); ++i) {
    r[i] = std::hypot(path[i].x, path[i].y);
    max_r = std::max(max_r, r[i]);
  }

  const int nl = std::min(n, 5);
  double lateral = 0.0;
  for (int i = 0; i < nl; ++i) lateral += std::fabs(path[static_cast<std::size_t>(i)].y);
  lateral /= nl;

  double chord = std::max(p_.lane_change_min_len, p_.lane_change_factor * lateral);
  chord = std::min(chord, 0.8 * max_r);

  int j = n - 1;
  for (int i = 0; i < n; ++i) {
    if (r[static_cast<std::size_t>(i)] >= chord) {
      j = i;
      break;
    }
  }

  const int k = 5;
  const int lo = std::max(j - k, 0);
  const int hi = std::min(j + k, n - 1);
  double tx = path[static_cast<std::size_t>(hi)].x - path[static_cast<std::size_t>(lo)].x;
  double ty = path[static_cast<std::size_t>(hi)].y - path[static_cast<std::size_t>(lo)].y;
  const double tn = std::max(std::hypot(tx, ty), 1e-6);
  tx /= tn;
  ty /= tn;

  const Point2 p3 = path[static_cast<std::size_t>(j)];
  const double handle = p_.bezier_handle * std::hypot(p3.x, p3.y);
  const Point2 p1{handle, 0.0};
  const Point2 p2{p3.x - handle * tx, p3.y - handle * ty};

  constexpr int kSamples = 200;
  Vec bx(kSamples), by(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    const double u = static_cast<double>(i) / (kSamples - 1);
    const double b = 3 * (1 - u) * (1 - u) * u;
    const double c = 3 * (1 - u) * u * u;
    const double d = u * u * u;
    bx[static_cast<std::size_t>(i)] = b * p1.x + c * p2.x + d * p3.x;  // p0 is the origin
    by[static_cast<std::size_t>(i)] = b * p1.y + c * p2.y + d * p3.y;
  }

  Vec arclen(kSamples, 0.0);
  for (int i = 1; i < kSamples; ++i) {
    const auto s = static_cast<std::size_t>(i);
    arclen[s] = arclen[s - 1] + std::hypot(bx[s] - bx[s - 1], by[s] - by[s - 1]);
  }
  for (int i = 0; i < kSamples; ++i) arclen[static_cast<std::size_t>(i)] += i * 1e-7;

  const int m = std::max(static_cast<int>(arclen.back() / p_.path_spacing) + 1, 2);
  const Vec query = linspace(0.0, arclen.back(), m);
  const Vec qx = interpolate(query, arclen, bx);
  const Vec qy = interpolate(query, arclen, by);

  Path out;
  out.reserve(static_cast<std::size_t>(m - 1) + path.size() - static_cast<std::size_t>(j));
  for (int i = 0; i < m - 1; ++i) out.push_back({qx[static_cast<std::size_t>(i)], qy[static_cast<std::size_t>(i)]});
  out.insert(out.end(), path.begin() + j, path.end());
  return out;
}

// ---------------------------------------------------------------------------------------------
// Savitzky-Golay
// ---------------------------------------------------------------------------------------------

const CenterlineGenerator::SGKernel& CenterlineGenerator::sgKernel(int window, int deriv) {
  const auto key = std::make_pair(window, deriv);
  auto it = sg_cache_.find(key);
  if (it != sg_cache_.end()) return it->second;

  const int half = window / 2;
  const int order = sg_polyorder_;
  Vec offsets(static_cast<std::size_t>(window));
  for (int i = 0; i < window; ++i) offsets[static_cast<std::size_t>(i)] = i - half;

  SGKernel k;

  // Interior kernel: derivative `deriv` of the least-squares polynomial at the window centre.
  const std::vector<Vec> pinv = vanderPinv(offsets, order + 1);
  const double fact = factorial(deriv);
  k.interior.resize(static_cast<std::size_t>(window));
  for (int i = 0; i < window; ++i)
    k.interior[static_cast<std::size_t>(i)] = pinv[static_cast<std::size_t>(deriv)][static_cast<std::size_t>(i)] * fact;

  // Edge matrices: evaluate the polynomial fitted to the first/last `window` samples at the samples
  // that lack a full window. Slope at the ends comes from a straight-line fit.
  const int edge_order = (deriv == 1) ? std::min(order, 1) : order;
  const std::vector<Vec> edge_pinv = vanderPinv(offsets, edge_order + 1);  // (edge_order+1) x window

  auto evalRows = [&](int tau) {
    Vec row(static_cast<std::size_t>(order + 1), 0.0);
    if (deriv == 0) {
      double v = 1.0;
      for (int q = 0; q <= order; ++q) {
        row[static_cast<std::size_t>(q)] = v;
        v *= tau;
      }
    } else {
      double v = 1.0;  // tau^(q-1)
      for (int q = 1; q <= order; ++q) {
        row[static_cast<std::size_t>(q)] = q * v;
        v *= tau;
      }
    }
    return row;
  };

  auto build = [&](int tau_begin, int tau_end) {
    Vec m(static_cast<std::size_t>(half * window), 0.0);
    for (int tau = tau_begin; tau <= tau_end; ++tau) {
      const Vec row = evalRows(tau);
      const auto r = static_cast<std::size_t>(tau - tau_begin);
      for (int c = 0; c < window; ++c) {
        double s = 0.0;
        for (int q = 0; q <= edge_order; ++q)  // rows above edge_order are zero-padded
          s += row[static_cast<std::size_t>(q)] * edge_pinv[static_cast<std::size_t>(q)][static_cast<std::size_t>(c)];
        m[r * static_cast<std::size_t>(window) + static_cast<std::size_t>(c)] = s;
      }
    }
    return m;
  };
  k.left = build(-half, -1);
  k.right = build(1, half);

  return sg_cache_.emplace(key, std::move(k)).first->second;
}

CenterlineGenerator::Vec CenterlineGenerator::sgApply(const Vec& a, int window, int deriv) {
  const SGKernel& k = sgKernel(window, deriv);
  const int n = static_cast<int>(a.size());
  const int half = window / 2;
  Vec out(a.size());

  // conv1d (cross-correlation) over valid positions
  for (int i = 0; i + window <= n; ++i) {
    double s = 0.0;
    for (int q = 0; q < window; ++q) s += k.interior[static_cast<std::size_t>(q)] * a[static_cast<std::size_t>(i + q)];
    out[static_cast<std::size_t>(i + half)] = s;
  }
  if (window < 3) return out;

  for (int r = 0; r < half; ++r) {
    double head = 0.0, tail = 0.0;
    for (int c = 0; c < window; ++c) {
      head += k.left[static_cast<std::size_t>(r * window + c)] * a[static_cast<std::size_t>(c)];
      tail += k.right[static_cast<std::size_t>(r * window + c)] * a[static_cast<std::size_t>(n - window + c)];
    }
    out[static_cast<std::size_t>(r)] = head;
    out[static_cast<std::size_t>(n - half + r)] = tail;
  }
  return out;
}

CenterlineGenerator::Vec CenterlineGenerator::boxSmooth(const Vec& a, int w) {
  const int n = static_cast<int>(a.size());
  w = std::min(w, n);
  if (w % 2 == 0) --w;
  if (w < 3) return a;
  const int pad = w / 2;
  Vec out(a.size());
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int t = -pad; t <= pad; ++t) s += a[static_cast<std::size_t>(std::clamp(i + t, 0, n - 1))];  // replicate pad
    out[static_cast<std::size_t>(i)] = s / w;
  }
  return out;
}

CenterlineGenerator::Vec CenterlineGenerator::gradient(const Vec& a) {
  const std::size_t n = a.size();
  Vec g(n, 0.0);
  if (n < 2) return g;
  g[0] = a[1] - a[0];
  g[n - 1] = a[n - 1] - a[n - 2];
  for (std::size_t i = 1; i + 1 < n; ++i) g[i] = 0.5 * (a[i + 1] - a[i - 1]);
  return g;
}

// ---------------------------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------------------------

Result CenterlineGenerator::generate(const std::vector<Point2>& white,
                                     const std::vector<Point2>& yellow) {
  std::array<std::vector<Candidate>, 2> near, far, yel;

  if (!white.empty()) {
    const std::size_t n = white.size();
    Vec x(n), y(n);
    std::vector<int> bins(n);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = white[i].x;
      y[i] = white[i].y;
      bins[i] = distBin(x[i]);
    }

    const LineSplit ls = splitLines(x, y);
    const std::vector<char> line_is_left = labelLines(y, ls, bins);
    std::vector<char> pt_is_left(n);
    for (std::size_t i = 0; i < n; ++i) pt_is_left[i] = line_is_left[static_cast<std::size_t>(ls.line_id[i])];
    updateMemory(y, bins, pt_is_left);

    std::vector<Candidate> cands;
    for (std::size_t l = 0; l < ls.start.size(); ++l) {
      const auto s = static_cast<std::ptrdiff_t>(ls.start[l]);
      const auto c = static_cast<std::ptrdiff_t>(ls.count[l]);
      cands.clear();
      lineCenterline(Vec(x.begin() + s, x.begin() + s + c), Vec(y.begin() + s, y.begin() + s + c),
                     line_is_left[l] ? kWhiteLeft : kWhiteRight, cands);
      for (Candidate& cand : cands) {
        auto& dst = (cand.kind == Kind::Near ? near : far)[idx(cand.lane)];
        dst.push_back(std::move(cand));
      }
    }
  }

  if ((near[0].empty() || near[1].empty()) && !yellow.empty()) {
    std::vector<Candidate> cands;
    yellowCandidates(yellow, cands);
    for (Candidate& cand : cands) yel[idx(cand.lane)].push_back(std::move(cand));
  }

  // select(): near + far if any near exists, otherwise yellow + far.
  auto select = [&](Side s) {
    std::vector<const Candidate*> out;
    const auto i = idx(s);
    for (const Candidate& c : (near[i].empty() ? yel[i] : near[i])) out.push_back(&c);
    for (const Candidate& c : far[i]) out.push_back(&c);
    return out;
  };

  Result res;
  res.left = fuseSide(Side::Left, select(Side::Left));
  res.right = fuseSide(Side::Right, select(Side::Right));

  if (res.left.empty() && res.right.empty()) {
    failedFrame();
    return res;  // ok = false
  }

  bad_frame_streak_ = 0;
  res.ok = true;
  if (p_.lane_change_bezier) {
    // Applied after fuse so the tracked (EMA) memory stays raw.
    res.left = bezierLaneChange(res.left);
    res.right = bezierLaneChange(res.right);
  }
  // vision.py called anchor_to_origin() on the output; a no-op once the Bezier is applied.
  res.left = anchorToOrigin(res.left);
  res.right = anchorToOrigin(res.right);
  return res;
}

}  // namespace centerlinexx