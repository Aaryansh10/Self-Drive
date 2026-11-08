#pragma once

#include <limits>
#include <vector>
#include "path_generation_cpp/centerline_generator.hpp"
#include "path_generation_cpp/skeleton_map.hpp"

namespace centerline {

struct RoadTrackerParams {
  double boundary_tolerance = 2.0;   // metres, at the vehicle
  double tolerance_per_m = 0.10;     // allow for curvature and map uncertainty ahead
  double min_forward_span = 0.25;
  double min_forward_alignment = 0.55;
  double max_branch_heading_error = 0.7;  // radians from the tracked road
  double bin_size = 0.5;
  double reference_timeout = 8.0;
  double reference_max_distance = 12.0;
  double reference_max_heading_change = 0.9;
  double fallback_timeout = 2.0;
  double fallback_max_distance = 3.0;
  double fallback_max_heading_change = 0.5;
  double fallback_max_anchor_distance = 1.0;
};

// Road association and short-gap path continuity are kept in the map frame.
// The underlying generator is reset for each map, because its EMA is robot-relative.
class RoadTracker {
 public:
  RoadTracker(const Params& generator_params = Params(), const RoadTrackerParams& params = RoadTrackerParams());
  Result update(const std::vector<Point2>& white, const std::vector<Point2>& yellow,
                const RobotPose& robot, double stamp, int selected_lane);
  void reset();

 private:
  struct Reference {
    double x = 0, y = 0, yaw = 0, stamp = 0;
    bool valid = false;
  };
  struct Selection {
    std::vector<Point2> white, yellow;
    std::vector<Point2> center_samples;
  };
  Selection select(const std::vector<Point2>& white, const std::vector<Point2>& yellow,
                   double intercept, double slope) const;
  Path recover(const Path& world, const RobotPose& robot) const;
  static Path toWorld(const Path& local, const RobotPose& robot);

  CenterlineGenerator generator_;
  RoadTrackerParams p_;
  Reference road_;
  RobotPose observed_pose_{};
  double observed_stamp_ = 0;
  Path left_world_, right_world_;
  double last_stamp_ = -std::numeric_limits<double>::infinity();
};
}  // namespace centerline
