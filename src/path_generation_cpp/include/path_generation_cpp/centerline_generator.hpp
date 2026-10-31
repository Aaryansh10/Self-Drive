// Lane centerline generation (C++ port of vision_py/centerline.py::CenterlineGenerator).
//
// ROS-free on purpose: the node (centerline_node.cpp) only converts messages <-> Point2 vectors, and
// this class can be unit-tested / benchmarked standalone.
//
// Coordinates follow the ground frame used by vision.py: x forward, y left, metres.
#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace centerline {

struct Point2 {
  double x{0.0};
  double y{0.0};
};

// An empty Path means "no centerline for this side" (Python returned None).
using Path = std::vector<Point2>;

struct Params {
  double max_gap = 1.0;                  // gap (m) between consecutive points that starts a new line
  double bin_width = 0.5;                // longitudinal bin size (m) for lane/width memory
  int max_bad_frame_streak = 20;         // reset memory after this many failed frames
  double max_range = 8.0;
  double nominal_road_width = 7.4;
  double min_road_width = 6.6;
  double max_road_width = 8.2;
  int min_bin_points = 1;
  double max_width_update = 0.5;
  // Optional lateral-normal fallback; disabled unless BOTH are set (same as the Python default).
  std::optional<double> normal_blend_tx_min;
  std::optional<double> normal_blend_tx_max;
  double min_offset_fraction = 0.5;
  double yellow_gate_fraction = 0.25;
  double path_spacing = 0.1;
  double path_start = 0.2;
  int final_smooth_window = 25;
  double near_field_range = 3.5;
  int min_near_pts = 2;
  int width_stale_decay_frames = 30;
  double width_stale_decay_alpha = 0.05;
  int sg_polyorder = 2;
  double track_ema_alpha = 0.35;
  double width_ema_alpha = 0.15;
  double centerline_ema_alpha = 0.4;
  double width_update_max_x = 4.0;
  double resample_step = 0.1;
  double position_window_m = 1.0;
  double tangent_window_m = 2.5;
  double far_near_field_range = 4.0;
  bool lane_change_bezier = true;
  double lane_change_factor = 2.5;
  double lane_change_min_len = 3.0;
  double bezier_handle = 0.4;
};

struct Result {
  Path left;        // empty => no left centerline
  Path right;       // empty => no right centerline
  bool ok = false;  // false => failed frame (both sides missing)
};

enum class Side { Left = 0, Right = 1 };
enum class Kind { Near, Far, Yellow };

struct OffsetSpec {
  Side lane;
  double fraction;
  bool toward_right;
  Kind kind;
};

class CenterlineGenerator {
 public:
  using Vec = std::vector<double>;

  explicit CenterlineGenerator(const Params& params = Params());

  // Equivalent of generate_centerlines() followed by anchor_to_origin() on both outputs
  // (which is what vision.py did). `yellow` may be empty.
  Result generate(const std::vector<Point2>& white, const std::vector<Point2>& yellow);

  void resetMemory();

  const Params& params() const { return p_; }

 private:
  struct Candidate {
    Side lane;
    Kind kind;
    Vec x, y, radial;  // radial is strictly increasing
    double fraction;
  };

  struct LineSplit {
    std::vector<int> line_id;  // per point
    std::vector<int> start;    // first point of each line
    std::vector<int> count;    // points in each line
  };

  struct SGKernel {
    Vec interior;  // window
    Vec left;      // half x window, row-major
    Vec right;     // half x window, row-major
  };

  // memory
  void failedFrame();
  void updateMemory(const Vec& y, const std::vector<int>& bins, const std::vector<char>& pt_is_left);

  // lane splitting / labelling
  LineSplit splitLines(const Vec& x, const Vec& y) const;
  int distBin(double x) const;
  std::vector<char> labelLines(const Vec& y, const LineSplit& ls, const std::vector<int>& bins) const;

  // candidate generation
  int oddWindow(double meters, int n) const;
  bool resampleLine(const Vec& x, const Vec& y, Vec& rx, Vec& ry) const;
  void lineCenterline(Vec x, Vec y, const std::array<OffsetSpec, 2>& offsets,
                      std::vector<Candidate>& out);
  void yellowCandidates(const std::vector<Point2>& yellow, std::vector<Candidate>& out);
  std::vector<char> yellowGate(const Vec& x, const Vec& y) const;

  // fusion
  Path fuseSide(Side side, const std::vector<const Candidate*>& candidates);
  Vec maskedSmooth(const Vec& values, const std::vector<char>& valid, int window) const;
  static void lockNearField(Vec& x, Vec& y);
  Path bezierLaneChange(const Path& path) const;
  Path anchorToOrigin(const Path& path) const;

  // Savitzky-Golay
  const SGKernel& sgKernel(int window, int deriv);
  Vec sgApply(const Vec& a, int window, int deriv);
  static Vec boxSmooth(const Vec& a, int w);
  static Vec gradient(const Vec& a);

  void inwardNormal(double tx, double ty, bool toward_right, double& nx, double& ny) const;

  Params p_;
  int max_bins_ = 0;
  int num_out_ = 0;
  double min_offset_ = 0.0;
  int sg_polyorder_ = 2;
  int sg_min_window_ = 3;
  bool use_normal_blend_ = false;
  double ref_fraction_ = 0.25;  // smallest offset fraction in use; weight 1.0
  double taper_len_ = 1.0;      // metres over which a candidate's weight ramps in/out

  Vec d_grid_;
  std::map<std::pair<int, int>, SGKernel> sg_cache_;

  // tracked state (NaN = unknown)
  Vec prev_left_y_, prev_right_y_;
  Vec road_width_, width_stale_;
  std::array<Vec, 2> track_x_, track_y_;  // indexed by Side
  int bad_frame_streak_ = 0;
};

}  // namespace centerline