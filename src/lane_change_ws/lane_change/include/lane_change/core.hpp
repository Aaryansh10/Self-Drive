// Pure C++ (no ROS) lane-change planner + controller. Everything is 2-D, SI units.
// Base frame: x forward, y LEFT (same as the vision/IPM point clouds in link_base).
#pragma once
#include <string>
#include <vector>

namespace lane_change {

struct Pt {
  double x = 0.0, y = 0.0;
};
using Line = std::vector<Pt>;  // ordered polyline

struct Params {
  // ---- geometry (rules: lanes are 10 ft wide) ----
  double lane_width = 3.048;          // m
  double robot_width = 0.90;          // m (rules: 2..4 ft wide)
  double front_offset = 0.50;         // m, base frame origin -> front bumper
  double body_clear_margin = 0.10;    // m, extra clearance past the yellow line for the "fully in lane" event
  bool distance_is_from_base = true;  // goal.obstacle_distance measured from base origin (false: from front)
  // ---- rule window: lane change completed 10..13 ft before the obstacle ----
  double completion_distance = 3.505;  // m (11.5 ft, middle of the window)
  double window_min = 3.048;           // m (10 ft)
  double window_max = 3.962;           // m (13 ft)
  bool completion_at_center = false;   // false: event = body fully in new lane; true: event = on new centreline
  // ---- manoeuvre shape ----
  double nominal_length = 8.0;  // m of forward travel for the lateral shift
  double tracking_lag_s = 0.05;  // plan the 'fully in lane' event this many seconds (x speed) earlier: robot lags its command
  double line_smooth = 0.4;      // m, half-window of the moving-average smoothing of the target centreline (0 = off)
  double max_lat_accel = 1.2;   // m/s^2, bounds how short the manoeuvre may be
  bool allow_late_compress = true;
  // ---- speed (rules: 1 mph < v < 5 mph, test speed 3..5 mph) ----
  double speed_ratio = 0.85;
  double min_speed = 0.60;  // m/s (1.34 mph)
  double max_speed = 2.20;  // m/s (4.9 mph)
  double max_accel = 0.80;  // m/s^2
  // ---- pure pursuit ----
  double lookahead_min = 1.2, lookahead_gain = 0.6, lookahead_max = 3.0;  // m, s, m
  double max_wz = 1.2;                                                    // rad/s
  // ---- path output (for Nav2) ----
  double path_length = 10.0;      // m of path published ahead of the robot (transition + target centreline)
  double min_turn_radius = 0.0;   // m, 0 = off. If > 0 the transition is lengthened so its tightest curve has at least this radius
  // ---- success / safety ----
  double settle_lat_tol = 0.20;    // m
  double settle_yaw_tol = 0.14;    // rad
  double settle_distance = 0.8;    // m of travel inside the tolerances
  double settle_timeout = 12.0;    // s
  double min_lateral_shift = 1.5;  // m, target line closer than this => wrong/invalid line
  double max_shift_factor = 1.7;   // x lane_width: farther than this => invalid / diverged
  double lost_timeout = 2.0;       // s, abort if no fresh target line for this long
  double wait_timeout = 90.0;      // s waiting for the start point
  double max_change_time = 40.0;   // s
};

class Polyline {
 public:
  struct Proj {
    Pt p, t, n;                // closest point, unit tangent, unit LEFT normal
    double s = 0.0;            // arc length of the closest point
    double signed_dist = 0.0;  // (query - p) . n   (positive = query is left of the line)
  };
  // Needs >= 2 distinct points. Extends the line straight at both ends.
  bool build(const Line& raw, double extend_back, double extend_fwd, double smooth = 0.0);
  bool valid() const { return pts_.size() >= 2; }
  double length() const { return cum_.empty() ? 0.0 : cum_.back(); }
  Proj project(Pt q) const;
  void at(double s, Pt* p, Pt* t) const;

 private:
  std::vector<Pt> pts_;
  std::vector<double> cum_;
};

double smoother(double u);          // quintic smootherstep, 0..1
double smoother_inverse(double y);  // inverse of the above on [0,1]

enum class Phase { IDLE, WAITING, CHANGING, SETTLING, DONE, FAILED };
enum class Code {
  SUCCESS = 0, TOO_LATE = 1, LANE_LOST = 2, TIMEOUT = 3, TARGET_BLOCKED = 4,
  CANCELED = 5, INVALID_GOAL = 6, NO_DATA = 7, INVALID_TARGET_LINE = 8
};

struct GoalIn {
  double obstacle_distance = 0.0;    // m straight ahead along the lane at goal time
  double completion_distance = 0.0;  // m, 0 = Params default
  double target_speed = 0.0;         // m/s, 0 = automatic
  bool target_lane_free = true;
};
struct OdomIn {
  double x = 0, y = 0, yaw = 0, speed = 0;  // odom-frame pose, speed magnitude
};

struct StepOut {
  Phase phase = Phase::IDLE;
  bool command_active = false;  // true: publish v / wz; false: leave the robot to the lane follower
  double v = 0.0, wz = 0.0;
  bool finished = false;
  Code code = Code::SUCCESS;
  std::string message;
  double remaining_to_obstacle = 0.0, lateral_error = 0.0, progress = 0.0;
  Line path;  // path being tracked, base frame (for visualisation)
};

class Maneuver {
 public:
  explicit Maneuver(const Params& p) : P_(p) {}
  // Returns SUCCESS if the goal was accepted (phase -> WAITING)
  Code start(const GoalIn& goal, const OdomIn& od, double now, std::string* msg);
  // fresh_target_base: new target-lane centreline (base frame) or nullptr if nothing new arrived
  StepOut step(const OdomIn& od, const Line* fresh_target_base, double now);
  void cancel();

  Phase phase() const { return phase_; }
  double body_clear_distance() const { return clear_rec_ ? clear_dist_ : -1.0; }
  bool in_window() const {
    return clear_rec_ && clear_dist_ >= P_.window_min && clear_dist_ <= P_.window_max;
  }
  double distance_travelled() const { return p_total_; }
  double planned_length() const { return L_; }
  double target_speed() const { return v_target_; }  // m/s chosen for the manoeuvre (85% of entry speed, clamped)

 private:
  StepOut fail(Code c, const std::string& m, StepOut out);
  double l_min_for(double d) const;  // shortest allowed transition length for a lateral shift d
  double sigma_clear_for(double d0) const;
  double u_clear_for(double d0) const;
  static Line to_odom(const Line& b, const OdomIn& od);
  static Line to_base(const Line& o, const OdomIn& od);

  Params P_;
  Phase phase_ = Phase::IDLE;
  Code code_ = Code::SUCCESS;
  std::string message_;
  // goal
  double C_ = 0, C_eff_ = 0, D0_ = 0, v_target_ = 0, v_cmd_ = 0, t_goal_ = 0, t_change_ = 0, t_settle_ = 0, t_need_ = -1;
  double L_nom_ = 0, u_nom_ = 0;
  // plan
  double d0_ = 0, L_ = 0, u_clear_ = 0, sigma_clear_ = 0;
  // progress
  double p_total_ = 0, p_man_ = 0, settle_acc_ = 0, last_t_ = 0;
  bool have_last_ = false, have_prev_ = false;
  OdomIn prev_;
  // target line memory (dead reckoning when no fresh line)
  Line cache_odom_;
  double cache_time_ = -1e9;
  bool have_cache_ = false;
  // event bookkeeping
  bool clear_rec_ = false;
  double clear_dist_ = -1.0;
};

}  // namespace lane_change
