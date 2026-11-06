// Thread-safe bridge between the detection subscriber and the BT nodes.
#pragma once
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "bt/decision.hpp"

namespace bt {

class SharedState {
 public:
  // stale_timeout: if no detection list arrived for this long, the decision falls back to NONE
  SharedState(DecisionConfig cfg, double stale_timeout, std::function<double()> clock)
      : decider_(std::move(cfg)), stale_(stale_timeout), clock_(std::move(clock)) {}

  double now() const { return clock_(); }

  void on_detections(const std::vector<Detection>& dets) {
    const double t = clock_();
    std::lock_guard<std::mutex> l(m_);
    auto eligible = dets;
    // Obstacle memory may retain a pedestrian after live perception confirms clearance.
    if (have_lane_ && t - lane_msg_ <= stale_ && !pedestrian_in_lane_ &&
        t - lane_clear_since_ >= 0.5) {
      eligible.erase(std::remove_if(eligible.begin(), eligible.end(),
          [](const Detection& d) { return d.cls == "pedestrian"; }), eligible.end());
    }
    decision_ = decider_.update(eligible, t);
    t_msg_ = t;
    have_ = true;
  }

  void on_lane(bool pedestrian_in_lane) {
    const double t = clock_();
    std::lock_guard<std::mutex> l(m_);
    if (!have_lane_ || pedestrian_in_lane || t - lane_msg_ > stale_) lane_clear_since_ = t;
    pedestrian_in_lane_ = pedestrian_in_lane;
    lane_msg_ = t;
    have_lane_ = true;
  }

  Decision current() {
    const double t = clock_();
    std::lock_guard<std::mutex> l(m_);
    if (!have_ || t - t_msg_ > stale_) return Decision{};
    return decision_;
  }

  // a lane change for this exact decision succeeded
  void mark_handled(const Decision& d) {
    const double t = clock_();
    std::lock_guard<std::mutex> l(m_);
    decider_.mark_handled(d, t);
    if (decision_.action == d.action && decision_.cls == d.cls &&
        decision_.obstacle_id == d.obstacle_id && d.cls != "pedestrian") decision_ = Decision{};
  }

 private:
  std::mutex m_;
  Decider decider_;
  Decision decision_;
  double stale_;
  double t_msg_ = 0.0;
  bool have_ = false;
  bool have_lane_ = false, pedestrian_in_lane_ = true;
  double lane_msg_ = 0.0, lane_clear_since_ = 0.0;
  std::function<double()> clock_;
};

}  // namespace bt