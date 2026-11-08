#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bt {
struct StopLinePoint { double x, y; };
struct StopLineGeometry {
    bool valid = false;
    double distance = 0;
    int points = 0;
};

// Find a well-supported transverse ground-plane line x = intercept + slope*y.
// A bounded slope allows oblique views while excluding longitudinal markings.
inline StopLineGeometry find_stop_line(const std::vector<StopLinePoint>& points, int min_points,
                                       double min_span, double tolerance, double max_slope,
                                       double max_gap) {
    StopLineGeometry best;
    if (min_points < 2 || min_span <= 0 || tolerance <= 0 || max_slope < 0 || max_gap <= 0)
        return best;
    double min_y = 1e100, max_y = -1e100;
    for (const auto& p : points) if (std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0) {
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    if (max_y - min_y < min_span) return best;
    struct Sample { double intercept, x, y; };
    std::vector<Sample> samples;
    samples.reserve(points.size());
    constexpr int steps = 10;
    for (int k = 0; k <= steps; ++k) {
        const double slope = -max_slope + 2 * max_slope * k / steps;
        samples.clear();
        for (const auto& p : points) if (std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0)
            samples.push_back({p.x - slope * p.y, p.x, p.y});
        std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
            return a.intercept < b.intercept;
        });
        for (size_t first = 0, end = 0; first < samples.size(); ++first) {
            if (end < first) end = first;
            while (end < samples.size() && samples[end].intercept - samples[first].intercept <= 2*tolerance) ++end;
            if (end - first < static_cast<size_t>(min_points) || end - first <= static_cast<size_t>(best.points)) continue;
            std::vector<Sample> group(samples.begin() + first, samples.begin() + end);
            std::sort(group.begin(), group.end(), [](const Sample& a, const Sample& b) { return a.y < b.y; });
            for (size_t start = 0; start < group.size();) {
                size_t stop = start + 1;
                while (stop < group.size() && group[stop].y - group[stop-1].y <= max_gap) ++stop;
                if (stop - start >= static_cast<size_t>(min_points) &&
                    stop - start > static_cast<size_t>(best.points) &&
                    group[stop-1].y - group[start].y >= min_span) {
                    std::vector<double> forward;
                    for (size_t i = start; i < stop; ++i) forward.push_back(group[i].x);
                    const size_t edge = forward.size() / 10;
                    std::nth_element(forward.begin(), forward.begin() + edge, forward.end());
                    best = {true, forward[edge], static_cast<int>(forward.size())};
                }
                start = stop;
            }
        }
    }
    return best;
}
}  // namespace bt
