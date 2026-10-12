// Closed-loop check of the lane-change core WITHOUT ROS.
//   g++ -std=c++17 -O2 -Iinclude src/core.cpp tools/sim_check.cpp -o sim_check && ./sim_check
// Exit code = number of scenarios that missed their expectation (0 = all good).
//
// World: two 10 ft lanes, dashed yellow centre line, solid white outer lines. The robot starts in
// the right lane centred, drives as a perfect lane-follower until the core takes over, then follows
// the core's (v, wz) commands through a first-order-lag unicycle model. The core only sees odometry
// and the left-lane centreline in the robot frame (x in [1.5, 7.5] m, noisy, 10 Hz).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "lane_change/core.hpp"

using namespace lane_change;

struct Road {
  bool straight = true;
  double Rc = 0.0;  // radius of the yellow line (curves)
  double s = 1.0;   // +1 left curve, -1 right curve
  double lane_w = 3.048;
  double r_lane(int lane) const { return lane == 0 ? Rc + s * lane_w / 2 : Rc - s * lane_w / 2; }  // 0 right, 1 left
  double scale(int lane) const { return straight ? 1.0 : r_lane(lane); }
  void lane_point(double a, int lane, double* x, double* y, double* yaw, double lat_left = 0.0) const {
    if (straight) {
      *x = a;
      *y = (lane == 0 ? -lane_w / 2 : lane_w / 2) + lat_left;
      *yaw = 0.0;
    } else {
      double r = r_lane(lane) - s * lat_left;
      *x = r * std::sin(a);
      *y = s * (Rc - r * std::cos(a));
      *yaw = s * a;
    }
  }
  double along(double x, double y) const { return straight ? x : std::atan2(x, -s * (y - s * Rc)); }
  double offset_right(double x, double y) const {
    return straight ? -y : s * (std::hypot(x, y - s * Rc) - Rc);
  }
};

struct Scenario {
  std::string name;
  Road road;
  double v0;               // lane-following speed before the manoeuvre (m/s)
  double D_base;           // obstacle distance from base at trigger (m)
  double start_off = 0.0;  // initial lateral offset to the left (m)
  bool dropout = false;    // lose the centreline for 1 s mid-manoeuvre
  double noise = 0.02;     // m, lateral noise on centreline points
  bool at_center = false;
  bool expect_fail = false;
};

struct Result {
  Code code = Code::SUCCESS;
  std::string msg;
  double event = -1;
  double core_event = -1;
  bool white = false, hit = false;
  double lat_acc = 0, L = 0, time = 0, end_off = 0;
};

static Result run(const Scenario& sc, unsigned seed, Params P) {
  const Road& R = sc.road;
  P.completion_at_center = sc.at_center;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double sim_dt = 0.005, ctrl_dt = 0.05, line_dt = 0.1, tau = 0.15, robot_len_back = 0.7;

  double a = 0.0;
  double x, y, yaw;
  R.lane_point(a, 0, &x, &y, &yaw, sc.start_off);
  double v = sc.v0, wz = 0.0;
  const double scaleR = R.scale(0);
  double a_obs = a + sc.D_base / scaleR;

  Maneuver M(P);
  GoalIn g;
  g.obstacle_distance = sc.D_base;
  OdomIn od{x, y, yaw, v};
  std::string msg;
  Result res;
  Code c = M.start(g, od, 0.0, &msg);
  if (c != Code::SUCCESS) {
    res.code = c;
    res.msg = msg;
    return res;
  }

  bool dynamic = false;
  double t = 0.0, next_ctrl = 0.0, next_line = 0.0, t_dyn = 0.0;
  double cmd_v = v, cmd_wz = 0.0;
  bool ev = false;
  const Line* fresh = nullptr;
  Line line_buf;

  while (t < 80.0) {
    if (t >= next_ctrl - 1e-9) {
      next_ctrl += ctrl_dt;
      fresh = nullptr;
      bool drop = sc.dropout && dynamic && (t - t_dyn) > 1.5 && (t - t_dyn) < 2.5;
      if (t >= next_line - 1e-9 && !drop) {
        next_line += line_dt;
        line_buf.clear();
        double a_rb = R.along(x, y);
        double sc_l = R.scale(1);
        for (double aa = a_rb - 1.0 / sc_l; aa < a_rb + 14.0 / sc_l; aa += 0.04 / sc_l) {
          double px, py, pyaw;
          R.lane_point(aa, 1, &px, &py, &pyaw);
          double dx = px - x, dy = py - y;
          double bx = std::cos(yaw) * dx + std::sin(yaw) * dy;
          double by = -std::sin(yaw) * dx + std::cos(yaw) * dy;
          if (bx >= 1.5 && bx <= 7.5) line_buf.push_back(Pt{bx, by + sc.noise * nd(rng)});
        }
        fresh = &line_buf;
      }
      od = OdomIn{x, y, yaw, v};
      StepOut o = M.step(od, fresh, t);
      if (o.finished) {
        res.code = o.code;
        res.msg = o.message;
        break;
      }
      if (o.command_active) {
        if (!dynamic) {
          dynamic = true;
          t_dyn = t;
          wz = R.straight ? 0.0 : R.s * v / R.r_lane(0);
        }
        cmd_v = o.v;
        cmd_wz = o.wz;
      }
    }
    if (!dynamic) {
      a += v * sim_dt / scaleR;
      R.lane_point(a, 0, &x, &y, &yaw, sc.start_off);
    } else {
      v += (cmd_v - v) * sim_dt / tau;
      wz += (cmd_wz - wz) * sim_dt / tau;
      yaw += wz * sim_dt;
      x += v * std::cos(yaw) * sim_dt;
      y += v * std::sin(yaw) * sim_dt;
      double base_wz = R.straight ? 0.0 : R.s * v / R.r_lane(0);
      res.lat_acc = std::max(res.lat_acc, std::fabs(v * (wz - base_wz)));
    }
    double off = R.offset_right(x, y);
    const double w = P.robot_width;
    if (off + w / 2 > R.lane_w || off - w / 2 < -R.lane_w) res.white = true;
    double a_b = R.along(x, y);
    double d_base_obs = (a_obs - a_b) * scaleR;
    if (!ev && off + w / 2 <= 0.0) {
      ev = true;
      res.event = d_base_obs - P.front_offset;
    }
    if (d_base_obs - 0.3 < P.front_offset && d_base_obs + 0.3 > -robot_len_back) {
      double gap = (R.lane_w / 2 - 0.3) - (off + w / 2);
      if (gap < 0.0) res.hit = true;
    }
    t += sim_dt;
  }
  res.time = t;
  res.core_event = M.body_clear_distance();
  res.L = M.planned_length();
  res.end_off = R.offset_right(x, y) + R.lane_w / 2;
  return res;
}

int main() {
  Params P;
  const double mph = 0.44704;
  Road straight;
  Road lc;  // left curve, inside radius 10 m (rules: min inside radius 10 m)
  lc.straight = false;
  lc.Rc = 10.0 + 3.048;
  lc.s = 1.0;
  Road rc = lc;
  rc.s = -1.0;

  std::vector<Scenario> S = {
      {"straight 3mph  trig 20m", straight, 3 * mph, 20.0},
      {"straight 4mph  trig 14m", straight, 4 * mph, 14.0},
      {"straight 4mph  trig 12.5m", straight, 4 * mph, 12.5},
      {"straight 5mph  trig 30m", straight, 5 * mph, 30.0},
      {"straight 5mph  trig 12.5m", straight, 5 * mph, 12.5},
      {"straight 4mph  trig 10.5m (late)", straight, 4 * mph, 10.5},
      {"straight 4mph  trig 8m (compressed)", straight, 4 * mph, 8.0},
      {"straight 4mph  trig 6m (too late)", straight, 4 * mph, 6.0, 0.0, false, 0.02, false, true},
      {"straight 4mph  start 0.4m left", straight, 4 * mph, 15.0, 0.4},
      {"straight 4mph  lost line 1s", straight, 4 * mph, 15.0, 0.0, true},
      {"straight 4mph  noise 6cm", straight, 4 * mph, 15.0, 0.0, false, 0.06},
      {"left curve  4mph trig 15m", lc, 4 * mph, 15.0},
      {"right curve 4mph trig 15m", rc, 4 * mph, 15.0},
      {"left curve  5mph trig 25m", lc, 5 * mph, 25.0},
      {"right curve 3mph trig 13m", rc, 3 * mph, 13.0},
      {"straight 4mph  centre-event", straight, 4 * mph, 15.0, 0.0, false, 0.02, true},
  };

  std::printf("%-34s %-6s %-18s %-6s %-5s %-4s %-7s %-6s %-5s\n", "scenario (5 seeds)", "ok", "event dist m",
              "inWin", "white", "hit", "latacc", "L(m)", "t(s)");
  int bad = 0;
  for (const Scenario& sc : S) {
    int ok = 0, inwin = 0, white = 0, hit = 0;
    const int n = 5;
    double emin = 1e9, emax = -1e9, acc = 0, L = 0, tt = 0;
    std::string why;
    for (unsigned seed = 1; seed <= 5; ++seed) {
      Result r = run(sc, seed, P);
      if (r.code == Code::SUCCESS) ++ok; else why = r.msg.substr(0, 40);
      if (r.event > 0) {
        emin = std::min(emin, r.event);
        emax = std::max(emax, r.event);
        if (r.event >= P.window_min && r.event <= P.window_max) ++inwin;
      }
      white += r.white;
      hit += r.hit;
      acc = std::max(acc, r.lat_acc);
      L = r.L;
      tt = std::max(tt, r.time);
    }
    bool pass = sc.expect_fail ? (ok == 0)
                               : (ok == n && white == 0 && hit == 0 && (sc.at_center || inwin == n));
    if (!pass) ++bad;
    char ev[64];
    if (emin > 1e8) std::snprintf(ev, sizeof ev, "n/a");
    else std::snprintf(ev, sizeof ev, "%.2f..%.2f", emin, emax);
    std::printf("%-34s %d/%d    %-18s %d/%d    %-5d %-4d %-7.2f %-6.1f %-5.1f %s %s\n", sc.name.c_str(), ok, n, ev,
                inwin, n, white, hit, acc, L, tt, pass ? "PASS" : "FAIL", ok == n ? "" : why.c_str());
  }
  std::printf("\nrule window (front to obstacle when fully in the new lane): %.3f..%.3f m (10..13 ft)\n",
              P.window_min, P.window_max);
  std::printf("%d scenario(s) missed expectations\n", bad);
  return bad;
}
