#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "path_generation_cpp/centerline_generator.hpp"

namespace centerline {

struct MapGeometry {
  int width, height;
  double resolution, origin_x, origin_y, origin_yaw;
};
struct RobotPose { double x, y, yaw; };  // in the map header frame

// Convert cell centers to robot coordinates, crop to the forward planning region,
// then trace graph edges into separate ordered branches. Raster order is not lane order.
inline std::vector<Point2> skeletonMapPoints(const std::vector<int8_t>& cells,
    const MapGeometry& g, const RobotPose& robot, double max_x, double max_y,
    int threshold = 50) {
  if (g.width <= 0 || g.height <= 0 || !std::isfinite(g.resolution) ||
      g.resolution <= 0 || cells.size() != size_t(g.width) * g.height)
    throw std::invalid_argument("Invalid skeleton map geometry");
  const size_t count = cells.size();
  std::vector<Point2> points(count);
  std::vector<uint8_t> active(count, 0), visited(count, 0);
  std::vector<std::array<int, 8>> adjacent(count);
  std::vector<int> degree(count, 0);
  const double cg = std::cos(g.origin_yaw), sg = std::sin(g.origin_yaw);
  const double cr = std::cos(robot.yaw), sr = std::sin(robot.yaw);
  for (int y = 0; y < g.height; ++y) {
    for (int x = 0; x < g.width; ++x) {
      const size_t i = size_t(y) * g.width + x;
      if (cells[i] < threshold) continue;
      const double gx = (x + 0.5) * g.resolution, gy = (y + 0.5) * g.resolution;
      const double dx = g.origin_x + cg * gx - sg * gy - robot.x;
      const double dy = g.origin_y + sg * gx + cg * gy - robot.y;
      points[i] = {cr * dx + sr * dy, -sr * dx + cr * dy};
      active[i] = points[i].x >= 0 && points[i].x <= max_x && std::abs(points[i].y) <= max_y;
    }
  }
  const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  const int dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
  auto occupied = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < g.width && y < g.height && active[size_t(y) * g.width + x];
  };
  for (int y = 0; y < g.height; ++y) {
    for (int x = 0; x < g.width; ++x) {
      const int i = y * g.width + x;
      adjacent[i].fill(-1);
      if (!active[i]) continue;
      for (int d = 0; d < 8; ++d) {
        if (!occupied(x + dx[d], y + dy[d])) continue;
        // Avoid triangular shortcuts across a right-angle staircase.
        if (dx[d] && dy[d] && (occupied(x + dx[d], y) || occupied(x, y + dy[d]))) continue;
        adjacent[i][d] = (y + dy[d]) * g.width + x + dx[d];
        ++degree[i];
      }
    }
  }
  std::vector<Point2> result;
  int segment = 0;
  auto trace = [&](int start, int direction) {
    std::vector<Point2> branch{points[start]};
    int current = start, d = direction;
    while (d >= 0 && !(visited[current] & (1 << d))) {
      const int next = adjacent[current][d];
      visited[current] |= 1 << d;
      visited[next] |= 1 << ((d + 4) % 8);
      branch.push_back(points[next]);
      current = next;
      if (degree[current] != 2 || current == start) break;
      d = -1;
      for (int k = 0; k < 8; ++k)
        if (adjacent[current][k] >= 0 && !(visited[current] & (1 << k))) { d = k; break; }
    }
    if (branch.size() < 3) return;
    if (branch.front().x > branch.back().x) std::reverse(branch.begin(), branch.end());
    for (auto p : branch) { p.segment = segment; result.push_back(p); }
    ++segment;
  };
  // Endpoints and junctions first; remaining unvisited edges belong to loops.
  for (int pass = 0; pass < 2; ++pass)
    for (size_t i = 0; i < count; ++i) {
      if (!active[i] || (pass == 0 && degree[i] == 2)) continue;
      for (int d = 0; d < 8; ++d)
        if (adjacent[i][d] >= 0 && !(visited[i] & (1 << d))) trace(static_cast<int>(i), d);
    }
  return result;
}
}  // namespace centerline
