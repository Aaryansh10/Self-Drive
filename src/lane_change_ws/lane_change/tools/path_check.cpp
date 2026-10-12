// What does the published path look like? Straight road, robot in the right lane, target = left lane centreline.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include "lane_change/core.hpp"
using namespace lane_change;

static void run(const char* name, Params P, double v, double trig) {
  const double W = 3.048;
  Maneuver M(P);
  GoalIn g; g.obstacle_distance = trig;
  OdomIn od{0, -W / 2, 0, v};
  std::string msg;
  if (M.start(g, od, 0.0, &msg) != Code::SUCCESS) { std::printf("%s: start failed: %s\n", name, msg.c_str()); return; }
  for (double t = 0; t < 30; t += 0.05) {
    Line fresh;
    for (double x = 1.5; x <= 7.5; x += 0.04) fresh.push_back(Pt{x, W / 2 - od.y});  // left centreline, base frame
    StepOut o = M.step(od, &fresh, t);
    if (o.finished) { std::printf("%s: finished early: %s\n", name, o.message.c_str()); return; }
    if (o.phase == Phase::CHANGING && o.command_active) {
      const Line& p = o.path;
      double minR = 1e9, maxdh = 0, yend = p.back().y + od.y, len = 0, on_line_at = -1;
      for (size_t i = 1; i + 1 < p.size(); ++i) {
        double ax = p[i].x - p[i-1].x, ay = p[i].y - p[i-1].y, bx = p[i+1].x - p[i].x, by = p[i+1].y - p[i].y;
        double cr = ax * by - ay * bx, la = std::hypot(ax, ay), lb = std::hypot(bx, by);
        double kappa = std::fabs(cr) / (la * lb * std::hypot(ax + bx, ay + by) / 2 + 1e-12);
        if (kappa > 1e-6) minR = std::min(minR, 1.0 / kappa);
        maxdh = std::max(maxdh, std::fabs(std::atan2(cr, ax * bx + ay * by)));
      }
      for (size_t i = 1; i < p.size(); ++i) {
        len += std::hypot(p[i].x - p[i-1].x, p[i].y - p[i-1].y);
        if (on_line_at < 0 && std::fabs(p[i].y + od.y - W / 2) < 0.02) on_line_at = p[i].x;
      }
      std::printf("%-26s L=%.1f m | path starts (%.2f, %.2f) | on target line after x=%s m | ends y=%.3f (target %.3f) | min radius %.1f m | pts %zu, length %.1f m\n",
                  name, M.planned_length(), p.front().x, p.front().y, on_line_at < 0 ? "never" : (std::to_string(on_line_at).substr(0, 4)).c_str(), yend, W / 2, minR, p.size(), len);
      return;
    }
    od.x += v * 0.05;  // robot keeps driving straight until the manoeuvre starts
  }
  std::printf("%s: manoeuvre never started\n", name);
}

int main() {
  Params P;
  const double mph = 0.44704;
  run("default, 4 mph", P, 4 * mph, 20.0);
  P.min_turn_radius = 6.0;
  run("min_turn_radius 6 m, 4 mph", P, 4 * mph, 25.0);
  P.min_turn_radius = 10.0;
  run("min_turn_radius 10 m, 4 mph", P, 4 * mph, 30.0);
  return 0;
}
