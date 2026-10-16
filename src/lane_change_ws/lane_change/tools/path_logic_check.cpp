#include <cstdio>
#include "lane_change/path_logic.hpp"
using namespace lane_change;
static int fails = 0;
#define CHECK(name, cond) do { bool ok_ = (cond); std::printf("%-58s %s\n", name, ok_ ? "PASS" : "FAIL"); fails += !ok_; } while (0)
int main() {
  Poly straight; for (double x = 0.2; x < 15; x += 0.1) straight.push_back(P2{x, 0.0});
  std::string why;
  CHECK("straight 15 m path accepted", validate_path(straight, 0, 0, 3.0, 0.5, 3.0, &why));
  Poly wig; for (double x = 0.2; x < 15; x += 0.1) wig.push_back(P2{x, std::sin(2 * M_PI * x / 3.0)});
  bool okw = !validate_path(wig, 0, 0, 3.0, 0.5, 3.0, &why);
  CHECK("S-wiggle (1 m amplitude) rejected", okw && why.find("kink") != std::string::npos);
  Poly lc; for (double x = 0.0; x < 14; x += 0.1) { double u = std::min(1.0, x / 10.7); lc.push_back(P2{x, 3.048 * (u*u*u*(10 + u*(-15 + 6*u)))}); }
  CHECK("lane-change transition (R about 7 m) accepted", validate_path(lc, 0, 0, 3.0, 0.5, 3.0, &why));
  Poly sh; for (double x = 0.2; x < 2.0; x += 0.1) sh.push_back(P2{x, 0});
  CHECK("1.8 m path rejected (too short)", !validate_path(sh, 0, 0, 3.0, 0.5, 3.0, &why));
  CHECK("path 6 m beside the robot rejected", !validate_path(straight, 0, 6.0, 3.0, 0.5, 3.0, &why));
  Poly gap = straight; gap.erase(gap.begin() + 40, gap.begin() + 70);
  CHECK("path with a 3 m gap rejected", !validate_path(gap, 0, 0, 3.0, 0.5, 3.0, &why));
  // transform: robot at (10, 5) facing +y (yaw 90 deg); a point 2 m ahead of it (base 2,0) is at (10, 7)
  Poly b{{2.0, 0.0}}; Poly g = transform_poly(b, 10, 5, M_PI / 2);
  CHECK("base->odom transform (2 m ahead, yaw 90 deg)", std::fabs(g[0].x - 10) < 1e-9 && std::fabs(g[0].y - 7) < 1e-9);
  // lane chooser: sticky with margin
  LaneChooser lc2;
  int a = lc2.update(true, 0.1, true, 3.0, 0.3);
  int b2 = lc2.update(true, 1.6, true, 1.5, 0.3);   // other only 0.1 closer -> stay
  int c = lc2.update(true, 2.0, true, 0.9, 0.3);    // clearly closer -> switch
  int d = lc2.update(false, 0.0, true, 0.9, 0.3);   // current invalid -> take the valid one
  CHECK("lane chooser: picks nearest, sticky, switches, falls back", a == 0 && b2 == 0 && c == 1 && d == 1);
  std::printf("%d failure(s)\n", fails);
  return fails;
}
