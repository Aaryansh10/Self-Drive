#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include "path_generation_cpp/skeleton_map.hpp"

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  const centerline::MapGeometry g{120, 120, 0.1, -2., -6., 0.};
  std::vector<int8_t> white(g.width*g.height, 0), yellow(white.size(), 0);
  for (int x = 20; x < 100; ++x) {
    white[23*g.width+x] = white[97*g.width+x] = 100;
    yellow[60*g.width+x] = 100;
  }
  auto points = centerline::skeletonMapPoints(white, g, {0,0,0}, 8, 12);
  std::set<int> segments;
  for (size_t i = 0; i < points.size(); ++i) {
    segments.insert(points[i].segment);
    if (i && points[i].segment == points[i-1].segment)
      check(std::hypot(points[i].x-points[i-1].x, points[i].y-points[i-1].y) < .15,
            "consecutive lane points must follow neighboring cells");
  }
  check(segments.size() == 2, "parallel lanes must be distinct branches");
  check(std::abs(points.front().x - .05) < 1e-8, "use cell centers");
  centerline::CenterlineGenerator generator;
  auto result = generator.generate(points,
      centerline::skeletonMapPoints(yellow, g, {0,0,0}, 8, 12));
  check(result.ok && !result.left.empty() && !result.right.empty(), "map lanes must generate both paths");
  for (const auto& path : {result.left, result.right})
    for (const auto& p : path) check(std::isfinite(p.x) && std::isfinite(p.y), "finite centerline");

  generator.resetMemory();
  const auto yellow_only = generator.generate({},
      centerline::skeletonMapPoints(yellow, g, {0,0,0}, 8, 12));
  check(yellow_only.ok && !yellow_only.left.empty() && !yellow_only.right.empty(),
        "yellow map must support fallback when white is empty");

  // Rotating/translating both the grid and robot preserves the extracted local path.
  auto transformed = g;
  transformed.origin_x = 16; transformed.origin_y = 18; transformed.origin_yaw = std::acos(-1.)/2;
  const auto other = centerline::skeletonMapPoints(white, transformed,
                                                 {10,20,std::acos(-1.)/2}, 8, 12);
  check(points.size() == other.size(), "rotated map crop must match");
  for (size_t i = 0; i < points.size(); ++i)
    check(std::hypot(points[i].x-other[i].x, points[i].y-other[i].y) < 1e-8,
          "map and robot transforms must compose correctly");

  // Junctions must produce branches instead of interleaving raster rows.
  for (int y = 23; y <= 60; ++y) white[y*g.width+60] = 100;
  points = centerline::skeletonMapPoints(white, g, {0,0,0}, 8, 12);
  segments.clear();
  for (const auto& p : points) segments.insert(p.segment);
  check(segments.size() == 4, "T junction produces three branches plus opposite lane");
  generator.resetMemory();
  check(!generator.generate({}, {}).ok, "empty map clears paths");
  bool threw = false;
  try { centerline::skeletonMapPoints({}, g, {0,0,0}, 8, 12); }
  catch (const std::invalid_argument&) { threw = true; }
  check(threw, "malformed grid rejected");
  std::cout << "Map extraction and centerline checks passed\n";
}
