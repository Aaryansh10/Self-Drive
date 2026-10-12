#include "lane_change/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace lane_change {

namespace {
constexpr double kPeakSecondDeriv = 5.7735;  // max |sigma''| of the quintic smootherstep

inline double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }
inline double hypot2(double a, double b) { return std::sqrt(a * a + b * b); }
inline double wrap(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}
std::string fmt(const char* f, double a, double b = 0.0, double c = 0.0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c);
  return buf;
}
}  // namespace

// --------------------------------------------------------------------------- smoother step
double smoother(double u) {
  if (u <= 0.0) return 0.0;
  if (u >= 1.0) return 1.0;
  return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

double smoother_inverse(double y) {
  if (y <= 0.0) return 0.0;
  if (y >= 1.0) return 1.0;
  double lo = 0.0, hi = 1.0;
  for (int i = 0; i < 50; ++i) {
    double mid = 0.5 * (lo + hi);
    (smoother(mid) < y ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// --------------------------------------------------------------------------- Polyline
bool Polyline::build(const Line& raw, double extend_back, double extend_fwd, double smooth) {
  pts_.clear();
  cum_.clear();
  for (const Pt& p : raw) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
    if (pts_.empty() || hypot2(p.x - pts_.back().x, p.y - pts_.back().y) > 1e-3) pts_.push_back(p);
  }
  if (pts_.size() < 2) {
    pts_.clear();
    return false;
  }
  // moving-average smoothing (suppresses point noise so the tangent / heading is usable)
  if (smooth > 0.0 && pts_.size() >= 5) {
    std::vector<Pt> sm(pts_.size());
    for (size_t i = 0; i < pts_.size(); ++i) {
      double sx = 0, sy = 0;
      int cnt = 0;
      for (size_t j = i + 1; j-- > 0;) {
        if (hypot2(pts_[j].x - pts_[i].x, pts_[j].y - pts_[i].y) > smooth) break;
        sx += pts_[j].x; sy += pts_[j].y; ++cnt;
      }
      for (size_t j = i + 1; j < pts_.size(); ++j) {
        if (hypot2(pts_[j].x - pts_[i].x, pts_[j].y - pts_[i].y) > smooth) break;
        sx += pts_[j].x; sy += pts_[j].y; ++cnt;
      }
      sm[i] = Pt{sx / cnt, sy / cnt};
    }
    // keep the original end points' longitudinal extent; only lateral noise is removed
    pts_ = sm;
    std::vector<Pt> dd;
    for (const Pt& p : pts_)
      if (dd.empty() || hypot2(p.x - dd.back().x, p.y - dd.back().y) > 1e-3) dd.push_back(p);
    pts_ = dd;
    if (pts_.size() < 2) return false;
  }
  // tangent estimates over >= 0.5 m to be robust against noise
  auto direction = [&](bool from_start) {
    const size_t n = pts_.size();
    Pt a = from_start ? pts_.front() : pts_.back();
    Pt b = a;
    for (size_t k = 1; k < n; ++k) {
      b = from_start ? pts_[k] : pts_[n - 1 - k];
      if (hypot2(b.x - a.x, b.y - a.y) >= 0.5) break;
    }
    double dx = from_start ? b.x - a.x : a.x - b.x;
    double dy = from_start ? b.y - a.y : a.y - b.y;
    double l = hypot2(dx, dy);
    if (l < 1e-9) return Pt{1.0, 0.0};
    return Pt{dx / l, dy / l};
  };
  Pt t0 = direction(true), t1 = direction(false);
  Pt front{pts_.front().x - t0.x * extend_back, pts_.front().y - t0.y * extend_back};
  Pt back{pts_.back().x + t1.x * extend_fwd, pts_.back().y + t1.y * extend_fwd};
  if (extend_back > 0.0) pts_.insert(pts_.begin(), front);
  if (extend_fwd > 0.0) pts_.push_back(back);
  cum_.assign(pts_.size(), 0.0);
  for (size_t i = 1; i < pts_.size(); ++i)
    cum_[i] = cum_[i - 1] + hypot2(pts_[i].x - pts_[i - 1].x, pts_[i].y - pts_[i - 1].y);
  return true;
}

Polyline::Proj Polyline::project(Pt q) const {
  Proj best;
  double best_d2 = 1e18;
  for (size_t i = 0; i + 1 < pts_.size(); ++i) {
    const Pt& a = pts_[i];
    const Pt& b = pts_[i + 1];
    double dx = b.x - a.x, dy = b.y - a.y;
    double len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) continue;
    double u = clampd(((q.x - a.x) * dx + (q.y - a.y) * dy) / len2, 0.0, 1.0);
    Pt c{a.x + u * dx, a.y + u * dy};
    double d2 = (q.x - c.x) * (q.x - c.x) + (q.y - c.y) * (q.y - c.y);
    if (d2 < best_d2) {
      best_d2 = d2;
      double len = std::sqrt(len2);
      best.p = c;
      best.t = Pt{dx / len, dy / len};
      best.n = Pt{-best.t.y, best.t.x};
      best.s = cum_[i] + u * len;
    }
  }
  best.signed_dist = (q.x - best.p.x) * best.n.x + (q.y - best.p.y) * best.n.y;
  return best;
}

void Polyline::at(double s, Pt* p, Pt* t) const {
  s = clampd(s, 0.0, length());
  size_t i = 0;
  while (i + 2 < pts_.size() && cum_[i + 1] < s) ++i;
  const Pt& a = pts_[i];
  const Pt& b = pts_[i + 1];
  double len = cum_[i + 1] - cum_[i];
  double u = len > 1e-9 ? (s - cum_[i]) / len : 0.0;
  if (p) *p = Pt{a.x + u * (b.x - a.x), a.y + u * (b.y - a.y)};
  if (t) {
    double dx = b.x - a.x, dy = b.y - a.y, l = hypot2(dx, dy);
    *t = l > 1e-9 ? Pt{dx / l, dy / l} : Pt{1.0, 0.0};
  }
}

// --------------------------------------------------------------------------- Maneuver
Line Maneuver::to_odom(const Line& b, const OdomIn& od) {
  Line o;
  o.reserve(b.size());
  double c = std::cos(od.yaw), s = std::sin(od.yaw);
  for (const Pt& p : b) o.push_back(Pt{od.x + c * p.x - s * p.y, od.y + s * p.x + c * p.y});
  return o;
}

Line Maneuver::to_base(const Line& o, const OdomIn& od) {
  Line b;
  b.reserve(o.size());
  double c = std::cos(od.yaw), s = std::sin(od.yaw);
  for (const Pt& p : o) {
    double dx = p.x - od.x, dy = p.y - od.y;
    b.push_back(Pt{c * dx + s * dy, -s * dx + c * dy});
  }
  return b;
}

double Maneuver::l_min_for(double d) const {
  double l_acc = v_target_ * std::sqrt(kPeakSecondDeriv * std::fabs(d) / P_.max_lat_accel);
  double l_rad = P_.min_turn_radius > 0.0 ? std::sqrt(kPeakSecondDeriv * std::fabs(d) * P_.min_turn_radius) : 0.0;
  return std::max(l_acc, l_rad);  // curvature of the quintic peaks at 5.7735*d/L^2
}

double Maneuver::sigma_clear_for(double d0) const {
  if (P_.completion_at_center) return 1.0;
  double keep = P_.lane_width * 0.5 - P_.robot_width * 0.5 - P_.body_clear_margin;  // max |offset| for "body in new lane"
  return clampd(1.0 - keep / std::max(1e-6, std::fabs(d0)), 0.02, 0.98);
}

double Maneuver::u_clear_for(double d0) const {
  if (P_.completion_at_center) return 1.0;
  return smoother_inverse(sigma_clear_for(d0));
}

Code Maneuver::start(const GoalIn& g, const OdomIn& od, double now, std::string* msg) {
  auto say = [&](const std::string& m) {
    if (msg) *msg = m;
  };
  if (!(g.obstacle_distance > 0.0) || !std::isfinite(g.obstacle_distance)) {
    say("obstacle_distance_m must be a positive number");
    return Code::INVALID_GOAL;
  }
  if (!g.target_lane_free) {
    say("target lane reported busy -> not changing lanes");
    return Code::TARGET_BLOCKED;
  }
  C_ = g.completion_distance > 0.0 ? g.completion_distance : P_.completion_distance;
  D0_ = g.obstacle_distance - (P_.distance_is_from_base ? P_.front_offset : 0.0);
  if (D0_ <= C_) {
    say(fmt("obstacle is already only %.2f m ahead (front); the lane change must be completed %.2f m before it",
            D0_, C_));
    return Code::TOO_LATE;
  }

  // reset state
  phase_ = Phase::WAITING;
  code_ = Code::SUCCESS;
  message_.clear();
  p_total_ = p_man_ = settle_acc_ = 0.0;
  have_last_ = false;
  prev_ = od;
  have_prev_ = true;
  have_cache_ = false;
  cache_odom_.clear();
  cache_time_ = -1e9;
  clear_rec_ = false;
  clear_dist_ = -1.0;
  t_goal_ = now;
  t_need_ = -1.0;

  v_target_ = g.target_speed > 0.0 ? clampd(g.target_speed, P_.min_speed, P_.max_speed)
                                   : clampd(P_.speed_ratio * od.speed, P_.min_speed, P_.max_speed);
  v_cmd_ = od.speed;
  double l_min0 = l_min_for(P_.lane_width);
  C_eff_ = C_ + P_.tracking_lag_s * v_target_;
  L_nom_ = std::max(P_.nominal_length, 1.05 * l_min0);
  u_nom_ = u_clear_for(P_.lane_width);
  say("accepted");
  return Code::SUCCESS;
}

void Maneuver::cancel() {
  if (phase_ == Phase::DONE || phase_ == Phase::FAILED || phase_ == Phase::IDLE) return;
  phase_ = Phase::FAILED;
  code_ = Code::CANCELED;
  message_ = "canceled";
}

StepOut Maneuver::fail(Code c, const std::string& m, StepOut out) {
  phase_ = Phase::FAILED;
  code_ = c;
  message_ = m;
  out.phase = phase_;
  out.finished = true;
  out.command_active = false;
  out.code = c;
  out.message = m;
  return out;
}

StepOut Maneuver::step(const OdomIn& od, const Line* fresh, double now) {
  StepOut out;
  bool just_started = false;
  out.phase = phase_;
  if (phase_ == Phase::IDLE) return out;
  if (phase_ == Phase::DONE || phase_ == Phase::FAILED) {
    out.finished = true;
    out.code = code_;
    out.message = message_;
    return out;
  }

  double dt = have_last_ ? clampd(now - last_t_, 1e-3, 0.5) : 0.05;
  last_t_ = now;
  have_last_ = true;

  // ---- target line (fresh, or dead-reckoned from the last good one)
  Line cur_base;
  if (fresh && fresh->size() >= 2) {
    cache_odom_ = to_odom(*fresh, od);
    cache_time_ = now;
    have_cache_ = true;
    cur_base = *fresh;
  } else if (have_cache_) {
    cur_base = to_base(cache_odom_, od);
  }
  Polyline line;
  bool have_line = !cur_base.empty() && line.build(cur_base, 2.0, std::max({6.0, P_.path_length, 2.0 * L_nom_ + 4.0}), P_.line_smooth);
  double age = have_cache_ ? now - cache_time_ : 1e9;

  // ---- longitudinal progress along the lane (projection of odom displacement on the lane tangent)
  double dpl = 0.0;
  if (have_prev_) {
    double tx = 1.0, ty = 0.0;  // base-frame lane tangent
    if (have_line) {
      Polyline::Proj p0 = line.project(Pt{0.0, 0.0});
      tx = p0.t.x;
      ty = p0.t.y;
    }
    double c = std::cos(od.yaw), s = std::sin(od.yaw);
    double tox = c * tx - s * ty, toy = s * tx + c * ty;
    dpl = (od.x - prev_.x) * tox + (od.y - prev_.y) * toy;
  }
  prev_ = od;
  have_prev_ = true;
  p_total_ += dpl;
  double remaining = D0_ - p_total_;
  out.remaining_to_obstacle = remaining;

  // ============================================================== WAITING
  if (phase_ == Phase::WAITING) {
    out.phase = phase_;
    if (now - t_goal_ > P_.wait_timeout)
      return fail(Code::TIMEOUT, "timed out waiting to reach the lane-change start point", out);
    if (remaining <= C_)
      return fail(Code::TOO_LATE, "obstacle reached the completion distance before the lane change started", out);
    double need = C_eff_ + u_nom_ * L_nom_;  // remaining distance at which the manoeuvre must begin
    if (remaining > need) return out;    // lane follower keeps driving

    // time to start
    if (!have_line) {
      if (t_need_ < 0.0) t_need_ = now;
      if (now - t_need_ > 1.0) return fail(Code::NO_DATA, "no target-lane centreline received", out);
      return out;
    }
    Polyline::Proj pr = line.project(Pt{0.0, 0.0});
    d0_ = pr.signed_dist;
    if (std::fabs(d0_) < P_.min_lateral_shift)
      return fail(Code::INVALID_TARGET_LINE,
                  fmt("target centreline is only %.2f m from the robot - it is not the neighbouring lane's line",
                      std::fabs(d0_)), out);
    if (std::fabs(d0_) > P_.max_shift_factor * P_.lane_width)
      return fail(Code::INVALID_TARGET_LINE,
                  fmt("target centreline is %.2f m from the robot - too far to be the neighbouring lane",
                      std::fabs(d0_)), out);

    sigma_clear_ = sigma_clear_for(d0_);
    u_clear_ = u_clear_for(d0_);
    double l_min = l_min_for(d0_);
    double l_plan = std::min((remaining - C_eff_) / u_clear_, 2.0 * L_nom_);
    std::string note = "ok";
    if (l_plan >= l_min) {
      L_ = l_plan;
    } else if (P_.allow_late_compress && l_plan >= 0.75 * l_min) {
      L_ = l_plan;
      note = fmt("started late: manoeuvre compressed to %.1f m (comfortable minimum %.1f m)", l_plan, l_min);
    } else {
      return fail(Code::TOO_LATE,
                  fmt("too late: need >= %.1f m of obstacle distance at trigger (had %.1f m left)",
                      C_eff_ + u_clear_ * l_min, remaining), out);
    }
    message_ = note;
    phase_ = Phase::CHANGING;
    p_man_ = 0.0;
    t_change_ = now;
    just_started = true;
    out.phase = phase_;
    // fall through to CHANGING in the same step
  }

  // ============================================================== CHANGING / SETTLING
  if (now - t_change_ > P_.max_change_time) return fail(Code::TIMEOUT, "lane change took too long", out);
  if (!have_line) return fail(Code::NO_DATA, "no target-lane centreline", out);
  if (age > P_.lost_timeout) return fail(Code::LANE_LOST, "target-lane centreline lost", out);

  Polyline::Proj pr = line.project(Pt{0.0, 0.0});
  double sd = pr.signed_dist;  // robot offset from the target centreline
  if (std::fabs(sd) > P_.max_shift_factor * P_.lane_width)
    return fail(Code::INVALID_TARGET_LINE, "robot diverged from the target centreline", out);

  if (phase_ == Phase::CHANGING && !just_started && dpl > 0.0) p_man_ += dpl;
  double e_now = phase_ == Phase::CHANGING ? d0_ * (1.0 - smoother(p_man_ / L_)) : 0.0;
  out.lateral_error = sd - e_now;
  out.progress = phase_ == Phase::CHANGING ? clampd(p_man_ / L_, 0.0, 1.0) : 1.0;

  // rule event: robot fully in the new lane
  if (!clear_rec_) {
    bool ev = P_.completion_at_center
                  ? (std::fabs(sd) <= P_.settle_lat_tol && p_man_ > 0.5 * L_)
                  : (std::fabs(sd) <= P_.lane_width * 0.5 - P_.robot_width * 0.5 - P_.body_clear_margin);
    if (ev && p_man_ > 0.0) {
      clear_rec_ = true;
      clear_dist_ = remaining;
    }
  }

  // path to track (base frame): target line shifted by the remaining lateral offset schedule
  const double ds = 0.25;
  const double plen = std::max({P_.lookahead_max + 1.0, P_.path_length, L_ + 3.0});  // always reaches the target line
  const int n = static_cast<int>(std::ceil(plen / ds));
  out.path.clear();
  for (int k = 0; k <= n; ++k) {
    Pt p, t;
    line.at(pr.s + k * ds, &p, &t);
    double e = phase_ == Phase::CHANGING ? d0_ * (1.0 - smoother((p_man_ + k * ds) / L_)) : 0.0;
    out.path.push_back(Pt{p.x - t.y * e, p.y + t.x * e});
  }

  // speed: ramp towards the manoeuvre speed
  double dv = clampd(v_target_ - v_cmd_, -2.0 * P_.max_accel * dt, P_.max_accel * dt);
  v_cmd_ = std::max(0.0, v_cmd_ + dv);

  // pure pursuit
  double ld = clampd(P_.lookahead_min + P_.lookahead_gain * v_cmd_, P_.lookahead_min, P_.lookahead_max);
  Pt target = out.path.back();
  for (const Pt& p : out.path) {
    if (p.x > 0.1 && hypot2(p.x, p.y) >= ld) {
      target = p;
      break;
    }
  }
  double d2 = std::max(1e-6, target.x * target.x + target.y * target.y);
  double kappa = 2.0 * target.y / d2;
  out.v = v_cmd_;
  out.wz = clampd(kappa * v_cmd_, -P_.max_wz, P_.max_wz);
  out.command_active = true;

  // phase transitions
  if (phase_ == Phase::CHANGING && p_man_ >= L_) {
    phase_ = Phase::SETTLING;
    t_settle_ = now;
    settle_acc_ = 0.0;
  } else if (phase_ == Phase::SETTLING) {
    double herr = std::atan2(pr.t.y, pr.t.x);
    if (std::fabs(sd) <= P_.settle_lat_tol && std::fabs(wrap(herr)) <= P_.settle_yaw_tol)
      settle_acc_ += std::max(0.0, dpl);
    else
      settle_acc_ = 0.0;
    if (settle_acc_ >= P_.settle_distance) {
      phase_ = Phase::DONE;
      code_ = Code::SUCCESS;
      message_ = clear_rec_ ? "success" : "success (rule event not observed)";
      out.finished = true;
      out.command_active = false;
      out.code = code_;
      out.message = message_;
    } else if (now - t_settle_ > P_.settle_timeout) {
      return fail(Code::TIMEOUT, "could not settle on the target centreline", out);
    }
  }
  out.phase = phase_;
  return out;
}

}  // namespace lane_change
