#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "mapping_cpp/lane_mapper.hpp"
#include "mapping_cpp/skeleton.hpp"

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  constexpr int w = 60, h = 40;
  std::vector<int8_t> grid(w * h, 0);
  grid[0] = -1;
  for (int y = 15; y <= 21; ++y)
    for (int x = 0; x < 55; ++x) grid[y*w+x] = 100;
  grid[5*w+5] = 100;
  const auto skeleton = lane_mapper::skeletonize(grid, w, h);
  check(skeleton[0] == -1, "unknown cells must be preserved");
  check(skeleton[5*w+5] == 0, "isolated speckle must be removed");
  for (int x = 5; x < 45; ++x) {
    int count = 0;
    for (int y = 0; y < h; ++y) count += skeleton[y*w+x] == 100;
    check(count == 1, "thick stripe must become one cell wide");
  }
  check(lane_mapper::skeletonize(skeleton, w, h) == skeleton, "thinning must be idempotent");
  check(lane_mapper::skeletonize(std::vector<int8_t>(w*h, -1), w, h) ==
        std::vector<int8_t>(w*h, -1), "empty unknown map");

  // Translation and a 90-degree rotation must not change robot-relative evidence.
  lane_mapper::Params p;
  p.width = p.height = 100;
  p.resolution = 0.1f;
  p.roi_x_min = 0.5f; p.roi_x_max = 3.f;
  p.roi_half_y_near = p.roi_half_y_far = 1.f;
  p.range_ref = 5.f; p.sigma_per_m = 0;
  p.min_points_for_miss = 1;
  lane_mapper::LaneMapper origin(p), shifted(p), rotated(p), yellow(p);
  const std::vector<lane_mapper::Point2> points{{1.5f, 0.5f}};
  for (int i = 0; i < 8; ++i) {
    origin.insert(points, {0, 0, 0}, 1);
    shifted.insert(points, {40, 30, 0}, 1);
    rotated.insert(points, {40, 30, std::acos(-1.)/2}, 1);
    yellow.insert({}, {0, 0, 0}, 1);
  }
  check(origin.isOccupied(1.55, .55), "white evidence must accumulate");
  check(!yellow.isOccupied(1.55, .55), "white must not contaminate yellow");
  const float initial = origin.logOdds(1.55, .55);
  for (int i = 0; i < 5; ++i) {
    origin.insert({}, {0, 0, 0}, 1);
    shifted.insert({}, {40, 30, 0}, 1);
    rotated.insert({}, {40, 30, std::acos(-1.)/2}, 1);
  }
  check(origin.logOdds(1.55, .55) < initial, "observed misses must decay occupancy");
  check(std::abs(origin.logOdds(1.55, .55) - shifted.logOdds(41.55, 30.55)) < 1e-3,
        "rolling bounds must use map origin");
  check(std::abs(origin.logOdds(1.55, .55) - rotated.logOdds(39.45, 31.55)) < 1e-3,
        "ROI evidence must rotate with robot");
  std::vector<int8_t> snapshot;
  lane_mapper::Window window;
  origin.snapshot(snapshot, window);
  check(std::count(snapshot.begin(), snapshot.end(), 100) > 0, "speckle filter retains supported cells");
  std::cout << "Lane map and skeleton checks passed\n";
}
