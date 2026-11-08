#include "path_generation_cpp/road_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace centerline {
namespace {
constexpr double kPi = 3.14159265358979323846;
double angleDiff(double a, double b) { return std::remainder(a - b, 2 * kPi); }
double distance(const RobotPose& pose, double x, double y) { return std::hypot(pose.x - x, pose.y - y); }
struct Segment {
  std::vector<Point2> points;
  double score = 0;
  double offset = 0;
  int kind = 0;  // 0 right white, 1 left white, 2 yellow
};
}  // namespace

RoadTracker::RoadTracker(const Params& gen, const RoadTrackerParams& p) : generator_(gen), p_(p) {
  if (p.boundary_tolerance <= 0 || p.tolerance_per_m < 0 || p.min_forward_span <= 0 ||
      p.min_forward_alignment <= 0 || p.min_forward_alignment > 1 ||
      p.max_branch_heading_error <= 0 || p.max_branch_heading_error >= kPi / 2 || p.bin_size <= 0 ||
      p.reference_timeout <= 0 || p.reference_max_distance <= 0 ||
      p.reference_max_heading_change <= 0 || p.reference_max_heading_change >= kPi / 2 ||
      p.fallback_timeout < 0 || p.fallback_max_distance < 0 ||
      p.fallback_max_heading_change < 0 || p.fallback_max_anchor_distance <= 0)
    throw std::invalid_argument("Invalid road tracker parameters");
}

void RoadTracker::reset() {
  road_ = {};
  left_world_.clear(); right_world_.clear();
  last_stamp_ = -std::numeric_limits<double>::infinity();
  generator_.resetMemory();
}

RoadTracker::Selection RoadTracker::select(const std::vector<Point2>& white,
                                            const std::vector<Point2>& yellow,
                                            double intercept, double slope) const {
  const double half_width = generator_.params().nominal_road_width * 0.5;
  std::vector<Segment> segments;
  auto collect = [&](const std::vector<Point2>& input, bool is_yellow) {
    size_t first = 0;
    while (first < input.size()) {
      size_t end = first + 1;
      while (end < input.size() && input[end].segment == input[first].segment) ++end;
      if (end - first < 3) { first = end; continue; }
      double xmin = input[first].x, xmax = xmin, length = 0;
      for (size_t i = first + 1; i < end; ++i) {
        xmin = std::min(xmin, input[i].x); xmax = std::max(xmax, input[i].x);
        length += std::hypot(input[i].x - input[i-1].x, input[i].y - input[i-1].y);
      }
      if (xmax - xmin < p_.min_forward_span || length <= 0 ||
          (xmax - xmin) / length < p_.min_forward_alignment) { first = end; continue; }
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (size_t i = first; i < end; ++i) {
        sx += input[i].x; sy += input[i].y;
        sxx += input[i].x * input[i].x; sxy += input[i].x * input[i].y;
      }
      const double n = static_cast<double>(end - first), denom = n * sxx - sx * sx;
      if (denom <= 1e-9 || std::abs(std::atan((n * sxy - sx * sy) / denom) - std::atan(slope)) >
          p_.max_branch_heading_error) { first = end; continue; }
      // Associate the segment with one of this road's two outer boundaries or its divider.
      const int begin_kind = is_yellow ? 2 : 0, end_kind = is_yellow ? 3 : 2;
      int best = -1;
      double best_score = std::numeric_limits<double>::infinity();
      for (int kind = begin_kind; kind < end_kind; ++kind) {
        const double offset = kind == 0 ? -half_width : kind == 1 ? half_width : 0;
        double error = 0;
        size_t inliers = 0;
        for (size_t i = first; i < end; ++i) {
          const double residual = std::abs(input[i].y - (intercept + slope * input[i].x + offset));
          if (residual <= p_.boundary_tolerance + p_.tolerance_per_m * input[i].x) {
            error += residual; ++inliers;
          }
        }
        if (inliers < 3 || inliers * 2 < end - first) continue;
        const double score = error / inliers;
        if (score < best_score) { best = kind; best_score = score; }
      }
      if (best >= 0) {
        Segment segment;
        segment.kind = best;
        segment.offset = best == 0 ? -half_width : best == 1 ? half_width : 0;
        segment.score = best_score;
        segment.points.assign(input.begin() + first, input.begin() + end);
        segments.push_back(std::move(segment));
      }
      first = end;
    }
  };
  collect(white, false); collect(yellow, true);

  // If two branches compete at the same distance, use only the one that best
  // matches the tracked road. Independent bins allow dashed markings and junctions.
  std::map<std::pair<int, int>, size_t> winners;
  for (size_t s = 0; s < segments.size(); ++s) {
    for (const auto& point : segments[s].points) {
      const double residual = std::abs(point.y - (intercept + slope * point.x + segments[s].offset));
      if (residual > p_.boundary_tolerance + p_.tolerance_per_m * point.x) continue;
      const auto key = std::make_pair(segments[s].kind, static_cast<int>(point.x / p_.bin_size));
      auto it = winners.find(key);
      if (it == winners.end() || segments[s].score < segments[it->second].score) winners[key] = s;
    }
  }
  Selection result;
  int next_white_id = 0, next_yellow_id = 0;
  for (size_t s = 0; s < segments.size(); ++s) {
    auto& output = segments[s].kind == 2 ? result.yellow : result.white;
    int& next_id = segments[s].kind == 2 ? next_yellow_id : next_white_id;
    bool in_run = false;
    int id = -1;
    for (auto point : segments[s].points) {
      const auto key = std::make_pair(segments[s].kind, static_cast<int>(point.x / p_.bin_size));
      const auto winner = winners.find(key);
      const double residual = std::abs(point.y - (intercept + slope * point.x + segments[s].offset));
      if (winner == winners.end() || winner->second != s ||
          residual > p_.boundary_tolerance + p_.tolerance_per_m * point.x) {
        in_run = false; continue;
      }
      if (!in_run) { id = next_id++; in_run = true; }
      point.segment = id;
      point.boundary_side = segments[s].kind == 0 ? -1 : segments[s].kind == 1 ? 1 : 0;
      output.push_back(point);
      result.center_samples.push_back({point.x, point.y - segments[s].offset});
    }
  }
  // Reconnect compatible short/dashed pieces on the same boundary. The map
  // graph can split a perfectly valid lane at small junctions or gaps. Keeping
  // every piece as a separate candidate makes the generator discard short runs.
  auto join = [this](std::vector<Point2>& points) {
    std::vector<std::vector<Point2>> runs;
    for (const auto& point : points) {
      if (runs.empty() || runs.back().back().segment != point.segment) runs.emplace_back();
      runs.back().push_back(point);
    }
    std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return a.front().x < b.front().x; });
    std::vector<std::vector<Point2>> tracks;
    for (auto& run : runs) {
      size_t best = tracks.size();
      double best_gap = generator_.params().max_gap;
      for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& prev = tracks[i].back();
        const auto& next = run.front();
        if (prev.boundary_side != next.boundary_side || next.x < prev.x - 0.05) continue;
        const double gap = std::hypot(next.x - prev.x, next.y - prev.y);
        if (gap <= best_gap && std::abs(next.y - prev.y) <= 0.35 + 0.5 * std::max(0.0, next.x - prev.x)) {
          best = i; best_gap = gap;
        }
      }
      if (best == tracks.size()) tracks.push_back(std::move(run));
      else tracks[best].insert(tracks[best].end(), run.begin(), run.end());
    }
    points.clear();
    for (size_t i = 0; i < tracks.size(); ++i)
      for (auto point : tracks[i]) { point.segment = static_cast<int>(i); points.push_back(point); }
  };
  join(result.white);
  join(result.yellow);
  return result;
}

Path RoadTracker::toWorld(const Path& local, const RobotPose& robot) {
  Path world;
  world.reserve(local.size());
  const double c = std::cos(robot.yaw), s = std::sin(robot.yaw);
  for (const auto& p : local) world.push_back({robot.x + c*p.x - s*p.y, robot.y + s*p.x + c*p.y});
  return world;
}

Path RoadTracker::recover(const Path& world, const RobotPose& robot) const {
  Path local;
  const double c = std::cos(robot.yaw), s = std::sin(robot.yaw);
  for (const auto& p : world) {
    const double dx = p.x - robot.x, dy = p.y - robot.y;
    const Point2 q{c*dx + s*dy, -s*dx + c*dy};
    if (q.x >= 0 && q.x <= generator_.params().max_range) local.push_back(q);
  }
  if (local.size() < 3 || std::hypot(local.front().x, local.front().y) > p_.fallback_max_anchor_distance)
    return {};
  local.insert(local.begin(), {0, 0});
  return local;
}

Result RoadTracker::update(const std::vector<Point2>& white, const std::vector<Point2>& yellow,
                           const RobotPose& robot, double stamp, int selected_lane) {
  if (!std::isfinite(stamp) || !std::isfinite(robot.x) || !std::isfinite(robot.y) || !std::isfinite(robot.yaw))
    throw std::invalid_argument("Invalid robot pose or map timestamp");
  if (stamp < last_stamp_) reset();
  last_stamp_ = stamp;
  if (road_.valid && (stamp - road_.stamp > p_.reference_timeout ||
      distance(robot, road_.x, road_.y) > p_.reference_max_distance ||
      std::abs(angleDiff(robot.yaw, road_.yaw)) > p_.reference_max_heading_change)) road_.valid = false;

  double intercept, slope;
  if (road_.valid) {
    const double c = std::cos(robot.yaw), s = std::sin(robot.yaw);
    const double dx = road_.x - robot.x, dy = road_.y - robot.y;
    const double ref_x = c*dx + s*dy, ref_y = -s*dx + c*dy;
    slope = std::tan(angleDiff(road_.yaw, robot.yaw));
    intercept = ref_y - slope * ref_x;
  } else {
    // The selected lane is the vehicle's current lane. The road center (yellow
    // divider) lies a quarter of the road width to its left or right.
    intercept = (selected_lane == 2 ? -0.25 : 0.25) * generator_.params().nominal_road_width;
    slope = 0;
  }
  Selection selected = select(white, yellow, intercept, slope);
  generator_.resetMemory();
  Result result = generator_.generate(selected.white, selected.yellow);
  if (!result.ok) {
    // The lane selection can lag a completed maneuver, and a previously tracked
    // road can be stale just after a turn. Reacquire from ego geometry only when
    // multiple independent markings support the same road; a lone foreign line
    // must not be allowed to redefine the corridor.
    for (int lane : {selected_lane, selected_lane == 2 ? 1 : 2}) {
      const double ego_center = (lane == 2 ? -0.25 : 0.25) * generator_.params().nominal_road_width;
      Selection alternative = select(white, yellow, ego_center, 0);
      struct Bin { double sum[3] = {0, 0, 0}; int count[3] = {0, 0, 0}; };
      std::map<int, Bin> bins;
      for (const auto& p : alternative.white) {
        const int side = p.boundary_side > 0 ? 1 : 0;
        auto& bin = bins[static_cast<int>(p.x / p_.bin_size)];
        bin.sum[side] += p.y; ++bin.count[side];
      }
      for (const auto& p : alternative.yellow) {
        auto& bin = bins[static_cast<int>(p.x / p_.bin_size)];
        bin.sum[2] += p.y; ++bin.count[2];
      }
      bool supported = false;
      const auto& gp = generator_.params();
      for (const auto& entry : bins) {
        const auto& b = entry.second;
        const auto between = [](double value, double lo, double hi) { return value >= lo && value <= hi; };
        if (b.count[0] && b.count[1] &&
            between(b.sum[1]/b.count[1] - b.sum[0]/b.count[0], gp.min_road_width, gp.max_road_width))
          supported = true;
        if (b.count[2]) {
          const double yellow_y = b.sum[2]/b.count[2];
          if (b.count[0] && between(yellow_y - b.sum[0]/b.count[0], gp.min_road_width/2, gp.max_road_width/2))
            supported = true;
          if (b.count[1] && between(b.sum[1]/b.count[1] - yellow_y, gp.min_road_width/2, gp.max_road_width/2))
            supported = true;
        }
      }
      if (!supported) continue;
      generator_.resetMemory();
      Result candidate = generator_.generate(alternative.white, alternative.yellow);
      if (candidate.ok) { selected = std::move(alternative); result = std::move(candidate); break; }
    }
  }

  // Fit the road center from associated boundaries, not from neighboring roads.
  // Only update the reference when the generator actually produced a path.
  if (result.ok && selected.center_samples.size() >= 3) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto& p : selected.center_samples) { sx += p.x; sy += p.y; sxx += p.x*p.x; sxy += p.x*p.y; }
    const double n = static_cast<double>(selected.center_samples.size());
    const double denom = n*sxx - sx*sx;
    if (denom > 1e-6) {
      const double fitted_slope = std::clamp((n*sxy - sx*sy) / denom, -1.0, 1.0);
      const double fitted_y = (sy - fitted_slope*sx) / n + fitted_slope*2.0;
      const double c = std::cos(robot.yaw), s = std::sin(robot.yaw);
      road_ = {robot.x + c*2.0 - s*fitted_y, robot.y + s*2.0 + c*fitted_y,
               robot.yaw + std::atan(fitted_slope), stamp, true};
    }
  }

  const bool can_recover = stamp >= observed_stamp_ && stamp - observed_stamp_ <= p_.fallback_timeout &&
      std::hypot(robot.x - observed_pose_.x, robot.y - observed_pose_.y) <= p_.fallback_max_distance &&
      std::abs(angleDiff(robot.yaw, observed_pose_.yaw)) <= p_.fallback_max_heading_change;
  const bool fresh_left = !result.left.empty(), fresh_right = !result.right.empty();
  if (can_recover) {
    if (!fresh_left) result.left = recover(left_world_, robot);
    if (!fresh_right) result.right = recover(right_world_, robot);
  }
  // Cache only a pair generated together; fallback never extends its own lifetime.
  if (fresh_left && fresh_right) {
    left_world_ = toWorld(result.left, robot); right_world_ = toWorld(result.right, robot);
    observed_pose_ = robot; observed_stamp_ = stamp;
  }
  result.ok = !result.left.empty() || !result.right.empty();
  return result;
}
}  // namespace centerline
