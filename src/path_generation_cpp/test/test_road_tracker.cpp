#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "path_generation_cpp/road_tracker.hpp"

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<centerline::Point2> lines(std::initializer_list<double> ys, double from = 0.1, double to = 7.8) {
  std::vector<centerline::Point2> out;
  int id = 0;
  for (double y : ys) {
    for (double x = from; x <= to; x += .1) out.push_back({x, y, id});
    ++id;
  }
  return out;
}

double lateralAt(const centerline::Path& path, double target) {
  check(!path.empty(), "missing path");
  const centerline::Point2* best = &path.front();
  for (const auto& p : path) if (std::abs(p.x - target) < std::abs(best->x - target)) best = &p;
  return best->y;
}

int main() {
  using namespace centerline;
  RoadTracker baseline, with_side_road;
  const RobotPose start{0, 0, 0};
  const auto white = lines({-1.85, 5.55});
  const auto yellow = lines({1.85});
  const Result reference = baseline.update(white, yellow, start, 1.0, 1);
  check(reference.ok && !reference.left.empty() && !reference.right.empty(), "acquire current road");
  check(std::abs(lateralAt(reference.right, 5)) < .6, "right lane follows robot's current lane");

  auto extra = lines({-9.25, -1.85, 5.55, 12.95});
  // A perpendicular intersection arm is close to both road boundaries.
  for (double y = -6.; y < 10.; y += .1) extra.push_back({4.0, y, 10});
  auto extra_yellow = lines({-5.55, 1.85, 9.25});
  const Result filtered = with_side_road.update(extra, extra_yellow, start, 1.0, 1);
  check(filtered.ok, "side road must not hide current road");
  check(std::abs(lateralAt(filtered.right, 5) - lateralAt(reference.right, 5)) < .2 &&
        std::abs(lateralAt(filtered.left, 5) - lateralAt(reference.left, 5)) < .2,
        "parallel and crossing roads must not move either centerline");

  // With only foreign road markings, they cannot become this road's boundaries.
  RoadTracker foreign_only;
  check(!foreign_only.update(lines({-9.25, 12.95}), lines({-5.55, 9.25}), start, 1, 1).ok,
        "foreign road alone must not generate paths");
  std::vector<Point2> cross;
  for (double y = -6.; y <= 8.; y += .1) cross.push_back({3., y, 0});
  check(!foreign_only.update(cross, {}, start, 1.1, 1).ok,
        "perpendicular markings must not generate paths");
  std::vector<Point2> oblique;
  for (double x = .1; x < 6.; x += .1) oblique.push_back({x, -1.85 + x, 0});
  check(!foreign_only.update(oblique, {}, start, 1.2, 1).ok,
        "oblique side road must not be mistaken for a current-road boundary");

  // Preserve the observed path in world coordinates as the robot advances through a gap.
  Result gap = baseline.update({}, {}, {1., 0., 0.}, 1.5, 1);
  check(gap.ok && !gap.right.empty(), "short marking gap retains forward path");
  check(std::abs(lateralAt(gap.right, 3.)) < .6, "fallback follows the world-frame road");
  check(!baseline.update({}, {}, {3.5, 0., 0.}, 4.0, 1).ok,
        "fallback expires instead of extending itself indefinitely");

  // Reacquire the same road even if both boundaries appear on the robot's left.
  Result reacquired = baseline.update(lines({.45, 7.85}), lines({4.15}), {3.6, -2.3, 0.}, 4.1, 1);
  check(reacquired.ok && !reacquired.right.empty(), "reacquire offset road after gap");
  const double right_y = lateralAt(reacquired.right, 5.);
  check(right_y > .45 && right_y < 4.15, "right lane stays inside its white/yellow boundaries");

  // A completed right-angle turn must discard the old heading and acquire the new road.
  const Result turned = baseline.update(white, yellow, {4., -2.3, 1.57079632679}, 4.2, 1);
  check(turned.ok && !turned.right.empty(), "reacquire road after a turn");
  check(std::abs(lateralAt(turned.right, 5)) < .6, "turned road uses new robot heading");

  RoadTracker bend;
  auto bend_white = lines({-1.85, 5.55}, .1, 4.0);
  auto bend_yellow = lines({1.85}, .1, 4.0);
  for (auto& p : bend_white) p.y -= .6 * p.x;
  for (auto& p : bend_yellow) p.y -= .6 * p.x;
  const auto bend_result = bend.update(bend_white, bend_yellow, start, 1., 1);
  check(bend_result.ok && !bend_result.right.empty(), "acquire a forward-bending road");
  const double bend_y = lateralAt(bend_result.right, 3.0);
  check(bend_y > -1.85 - .6 * 3.0 && bend_y < 1.85 - .6 * 3.0,
        "forward-bending road offsets must point inside the lane");

  // Map skeletons can be split into short dashed runs by gaps or junctions.
  RoadTracker fragmented;
  std::vector<Point2> dashed_white, dashed_yellow;
  int segment = 0;
  for (int n = 0; n < 10; ++n) {
    for (int j = 0; j < 5; ++j) {
      const double x = .4 + n*.7 + j*.1;
      dashed_white.push_back({x, -1.85, segment});
      dashed_yellow.push_back({x, 1.85, segment});
    }
    ++segment;
  }
  auto dashed_result = fragmented.update(dashed_white, dashed_yellow, start, 1., 1);
  check(dashed_result.ok && !dashed_result.right.empty(), "short dashed lane fragments generate a path");
  check(std::abs(lateralAt(dashed_result.right, 5)) < .6, "fragmented path stays in the lane");
  RoadTracker single_boundary;
  auto one = single_boundary.update(lines({-1.85}), {}, start, 1., 1);
  check(one.ok && !one.right.empty(), "one visible current-road edge still gives a centerline");
  check(std::abs(lateralAt(one.right, 5)) < .6, "single-edge path remains inside its lane");
  RoadTracker wrong_selection;
  auto recovered_selection = wrong_selection.update(white, yellow, start, 1., 2);
  check(recovered_selection.ok && !recovered_selection.right.empty(),
        "two consistent markings recover when selected lane is stale");
  check(std::abs(lateralAt(recovered_selection.right, 5)) < .6, "reacquisition stays inside the road");

  RoadTracker left_lane;
  const auto left_result = left_lane.update(lines({-5.55, 1.85}), lines({-1.85}), start, 1., 2);
  check(left_result.ok && std::abs(lateralAt(left_result.left, 5)) < .6,
        "left-lane selection seeds the correct road corridor");
  // A clock reset cannot reuse old paths.
  check(!baseline.update({}, {}, {4., -2.3, 1.57}, .1, 1).ok, "clock reset clears path memory");
  std::cout << "Road selection and intersection recovery checks passed\n";
}
