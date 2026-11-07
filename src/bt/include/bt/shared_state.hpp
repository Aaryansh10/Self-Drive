// Independent perception streams and a latched stop/intersection maneuver.
#pragma once
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
        obstacle_ = obstacles_.update(dets, obstacle_time_);
    }

    // Called once per sign-depth detection list, or per cloud when no sign is present.
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

    void mark_handled(const Decision& d) {
        std::lock_guard<std::mutex> lock(m_);
        obstacles_.mark_handled(d, now());
        if (obstacle_.action == d.action && obstacle_.obstacle_id == d.obstacle_id)
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
    std::function<double()> clock_;
    std::deque<uint8_t> turns_;
};
} // namespace bt
