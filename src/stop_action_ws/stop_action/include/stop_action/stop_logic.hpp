#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace igvc_stop
{
    struct Profile {
        double min_distance{0.0};
        double max_distance{0.0};
        double max_speed{0.0};
    };

    inline double target_gap(const Profile &p, double fraction) {
        return p.min_distance + std::clamp(fraction, 0.0, 1.0) * (p.max_distance - p.min_distance);
    }

    inline double trigger_distance(double v, double a, double t_lat, double gap) {
        v = std::max(v,0.0);
        return v*v/(2.0 * a) + v*t_lat + gap;
    }

    inline double reference_speed(double d, double v_meas, double a, double t_lat, double gap) {
        const double d_eff = d - std::max(v_meas,0.0) * t_lat - gap;
        if (d_eff <= 0.0)
        return 0.0;
        return std::sqrt(2.0*a*d_eff);
    }

    inline double next_command(double v_cmd_prev, double v_ref, double a_max, double dt) {
        const double lowest = v_cmd_prev - a_max*dt;
        return std::max(0.0, std::clamp(v_ref, lowest, v_cmd_prev));
    }

    inline double median(std::vector<double> v) {
        if (v.empty()) return 0.0;
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2.0);
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    }
}