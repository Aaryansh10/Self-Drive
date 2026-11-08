#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "bt/stop_line_geometry.hpp"

void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
int main() {
  using bt::StopLinePoint;
  auto detect = [](const std::vector<StopLinePoint>& pts) {
    return bt::find_stop_line(pts, 5, .6, .15, .5, .35);
  };
  std::vector<StopLinePoint> horizontal;
  for (int i = -10; i <= 10; ++i) horizontal.push_back({3.0 + .02 * std::sin(i), .1 * i});
  auto line = detect(horizontal);
  check(line.valid && line.points >= 20 && std::abs(line.distance - 3.) < .1, "valid horizontal line");
  std::vector<StopLinePoint> diagonal;
  for (int i = -10; i <= 10; ++i) diagonal.push_back({3. + .4 * i/10., .1 * i});
  check(detect(diagonal).valid, "moderate perspective angle accepted");
  std::vector<StopLinePoint> longitudinal;
  for (int i = 0; i < 60; ++i) longitudinal.push_back({1. + .1*i, .02*std::sin(i)});
  check(!detect(longitudinal).valid, "longitudinal points rejected");
  std::vector<StopLinePoint> scattered;
  for (int i = 0; i < 20; ++i) scattered.push_back({1. + .25*i, -.95 + .1*i});
  check(!detect(scattered).valid, "oblique scattered points rejected");
  std::vector<StopLinePoint> cluster;
  for (int i = 0; i < 30; ++i) cluster.push_back({2., .005*i});
  check(!detect(cluster).valid, "dense small blob rejected");
  std::vector<StopLinePoint> separated;
  for (int i = 0; i < 5; ++i) {
    separated.push_back({2., -.8 + .03*i});
    separated.push_back({2., .8 + .03*i});
  }
  check(!detect(separated).valid, "two separated blobs do not form a line");
  check(!detect({}).valid, "empty cloud rejected");
  std::cout << "Stop-line geometry checks passed\n";
}
