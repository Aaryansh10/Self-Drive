// Independent perception streams and a latched stop/intersection maneuver.
#pragma once
#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>
#include "bt/decision.hpp"

namespace bt {
class SharedState {
public:
    SharedState(DecisionConfig cfg, double stale_timeout, std::function<double()> clock)
        : obstacles_(rules_for(cfg, false)), intersections_(rules_for(cfg, true)),
          stale_(stale_timeout), clock_(std::move(clock)) {}

    double now() const { return clock_(); }

    void on_detections(const std::vector<Detection>& dets) {
        std::lock_guard<std::mutex> lock(m_);
        obstacle_time_ = now();
        auto eligible = dets;
        // Obstacle memory may retain a pedestrian after live perception confirms clearance.
        if (have_lane_ && obstacle_time_ - lane_msg_ <= stale_ && !pedestrian_in_lane_ &&
            obstacle_time_ - lane_clear_since_ >= 0.5) {
            eligible.erase(std::remove_if(eligible.begin(), eligible.end(),
                [](const Detection& d) { return d.cls == "pedestrian"; }), eligible.end());
        }
        obstacle_ = obstacles_.update(eligible, obstacle_time_);
    }

    void on_lane(bool pedestrian_in_lane) {
        std::lock_guard<std::mutex> lock(m_);
        const double t = now();
        if (!have_lane_ || pedestrian_in_lane || t - lane_msg_ > stale_) lane_clear_since_ = t;
        pedestrian_in_lane_ = pedestrian_in_lane;
        lane_msg_ = t;
        have_lane_ = true;
    }

    // Called once per stop-line cloud after geometric validation.
    void on_intersection_detections(const std::vector<Detection>& dets, bool reset_stop_sign = false) {
        std::lock_guard<std::mutex> lock(m_);
        if (reset_stop_sign) intersections_.clear_stop_sign();
        intersection_time_ = now();
        intersection_ = intersections_.update(dets, intersection_time_);
    }

    Decision current() {
        std::lock_guard<std::mutex> lock(m_);
        return current_locked();
    }

    Decision begin_intersection() {
        std::lock_guard<std::mutex> lock(m_);
        if (active_.action != Action::INTERSECTION) {
            const auto d = current_locked();
            if (d.action == Action::INTERSECTION) active_ = d;
        }
        return active_;
    }

    Decision active_intersection() {
        std::lock_guard<std::mutex> lock(m_);
        return active_;
    }

    void mark_intersection_stopped() {
        std::lock_guard<std::mutex> lock(m_);
        active_.needs_stop = false;
    }

    void complete_intersection(const Decision& sent) {
        std::lock_guard<std::mutex> lock(m_);
        intersections_.mark_handled(sent, now());
        last_completed_ = now();
        active_ = intersection_ = Decision{};
        if (!turns_.empty()) turns_.pop_front();
    }

    double last_intersection_completed() {
        std::lock_guard<std::mutex> lock(m_);
        return last_completed_;
    }

    // a lane change / stop for this exact decision succeeded
    void mark_handled(const Decision& d) {
        std::lock_guard<std::mutex> lock(m_);
        obstacles_.mark_handled(d, now());
        if (obstacle_.action == d.action && obstacle_.cls == d.cls &&
            obstacle_.obstacle_id == d.obstacle_id && d.cls != "pedestrian")
            obstacle_ = Decision{};
    }

    void set_turns(const std::vector<uint8_t>& turns) {
        std::lock_guard<std::mutex> lock(m_);
        turns_.assign(turns.begin(), turns.end());
    }

    bool peek_turn(uint8_t& turn) {
        std::lock_guard<std::mutex> lock(m_);
        if (turns_.empty()) return false;
        turn = turns_.front();
        return true;
    }

private:
    static DecisionConfig rules_for(DecisionConfig cfg, bool intersection) {
        // Stop lines have no tracker IDs, but their odom anchors are persistent.
        if (intersection) cfg.pseudo_ttl = cfg.handled_ttl;
        for (auto it = cfg.classes.begin(); it != cfg.classes.end();) {
            const bool is_intersection = it->second.action == Action::INTERSECTION || it->second.mark_stop_sign;
            if (is_intersection != intersection) it = cfg.classes.erase(it);
            else ++it;
        }
        return cfg;
    }

    Decision current_locked() const {
        const double t = now();
        const bool fresh_obstacle = t - obstacle_time_ <= stale_;
        if (fresh_obstacle && obstacle_.action == Action::STOP) return obstacle_;
        // Survives stop-line disappearance, a long stop, and pedestrian preemption.
        if (active_.action == Action::INTERSECTION) return active_;
        if (t - intersection_time_ <= stale_ && intersection_.action == Action::INTERSECTION)
            return intersection_;
        return fresh_obstacle ? obstacle_ : Decision{};
    }

    std::mutex m_;
    Decider obstacles_, intersections_;
    Decision obstacle_, intersection_, active_;
    double stale_;
    double obstacle_time_ = -1e9, intersection_time_ = -1e9, last_completed_ = -1e9;
    bool have_lane_ = false, pedestrian_in_lane_ = true;
    double lane_msg_ = 0.0, lane_clear_since_ = 0.0;
    std::function<double()> clock_;
    std::deque<uint8_t> turns_;
};
} // namespace bt