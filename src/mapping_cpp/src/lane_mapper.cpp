#include "mapping_cpp/lane_mapper.hpp"

#include <algorithm>
#include <cmath>

namespace lane_mapper {
    namespace {
        inline int pmod(int a, int n) {
            int m = a % n;
            return m < 0 ? m + n : m;
        }
    }

    LaneMapper::LaneMapper(const Params& p)
        : parameters_(p), width_(parameters_.width), height_(parameters_.height),
          log_odds_(size_t(width_) * height_, 0.f), frame_ev_(size_t(width_) * height_, 0.f),
          seen_(size_t(width_) * height_, 0) {
        touched_.reserve(8192);
    }

    void LaneMapper::reset() {
        std::fill(log_odds_.begin(), log_odds_.end(), 0.f);
        std::fill(seen_.begin(), seen_.end(), 0);
        initialised_ = false;
    }

    size_t LaneMapper::getIdx(int grid_x, int grid_y) const {
        return size_t(pmod(grid_y, height_) * width_ + pmod(grid_x, width_));
    }

    int LaneMapper::toGridNumber(double v) const {
        return int(std::floor(v / parameters_.resolution));
    }

    bool LaneMapper::inWindow(int grid_x, int grid_y) const {
        return grid_x >= origin_x_ && grid_x < origin_x_ + width_ 
            && grid_y >= origin_y_ && grid_y < origin_y_ + height_;
    }

    void LaneMapper::applyLogOdds(size_t i, float d) {
        log_odds_[i] = std::clamp(log_odds_[i] + d, parameters_.l_min, parameters_.l_max);
        seen_[i] = 1;
    }

    void LaneMapper::clearCol(int grid_x) {
        const int c = pmod(grid_x, width_);
        for (int r = 0; r < height_; ++r) {
            log_odds_[size_t(r) * width_ + c] = 0.f;
            seen_[size_t(r) * width_ + c] = 0;
        }        
    }

    void LaneMapper::clearRow(int grid_y) {
        const size_t base = size_t(pmod(grid_y, height_)) * width_;
        std::fill(log_odds_.begin() + base, log_odds_.begin() + base + width_, 0.f);
        std::fill(seen_.begin() + base, seen_.begin() + base + width_, 0);
    }

    void LaneMapper::recenter(const Pose2& pose) {
        // Calculate the new starting point of the map
        const int new_x = toGridNumber(pose.x) - width_ / 2;
        const int new_y = toGridNumber(pose.y) - height_ / 2;

        if (!initialised_) {
            reset();
            origin_x_ = new_x;
            origin_y_ = new_y;
            initialised_ = true;
            return;
        }

        const int dx = new_x - origin_x_;
        const int dy = new_y - origin_y_;

        if (std::abs(dx) >= width_ || std::abs(dy) >= height_) {
            std::fill(log_odds_.begin(), log_odds_.end(), 0.f);
            std::fill(seen_.begin(), seen_.end(), 0);
        } 
        else {
            const int clear_x_0 = std::min(origin_x_, new_x);
            const int clear_y_0 = std::min(origin_y_, new_y);
            for (int k = 0; k < std::abs(dx); ++k) {
                clearCol(clear_x_0 + k);
            }
            for (int k = 0; k < std::abs(dy); k++) {
                clearRow(clear_y_0 + k);
            }
        }

        origin_x_ = new_x;
        origin_y_ = new_y;
    }

    void LaneMapper::insert(const std::vector<Point2>& pts, const Pose2& pose, int frame_hp) {
        recenter(pose);

        const double sin_yaw = std::sin(pose.yaw);
        const double cos_yaw = std::cos(pose.yaw);


        // Calculates the evidences for all the cells of the map using gaussian splatting on each cell
        for (const auto& pt : pts) {
            // Calculating the deviation and the peak weight for the 2D Gaussian splatting
            const float radial_dist = std::hypot(pt.x, pt.y);
            const float sigma = parameters_.sigma_base + parameters_.sigma_per_m * radial_dist + 
                                parameters_.sigma_per_m2 * radial_dist * radial_dist;
            const float weight = 1.f / (1.f + (radial_dist / parameters_.range_ref) * (radial_dist / parameters_.range_ref));

            // Calculating the global coordinates (odom frame) of the point 
            const double global_x = pose.x + cos_yaw * pt.x - sin_yaw * pt.y;
            const double global_y = pose.y + sin_yaw * pt.x + cos_yaw * pt.y;

            // Store the center of the gaussian splat
            const int grid_x_center = toGridNumber(global_x);
            const int grid_y_center = toGridNumber(global_y);

            // Calculates the radius of the box for gaussian splat
            const int radius = int(std::ceil(3.f * sigma / parameters_.resolution));
            
            // Precalculates (2*sigma^2)^-1 for saving computation
            const float inverse_2_sigma_square = 1.f / (2.f * sigma * sigma);

            // Calculates the evidence for each cell in the gaussian splat
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int grid_x = grid_x_center + dx;
                    const int grid_y = grid_y_center + dy;

                    if (!inWindow(grid_x, grid_y)) 
                        continue;
                    
                    const double error_x = (grid_x + 0.5) * parameters_.resolution - global_x;
                    const double error_y = (grid_y + 0.5) * parameters_.resolution - global_y;
                    const float evidence = weight * std::exp(
                        -float(error_x * error_x + error_y * error_y) * inverse_2_sigma_square
                    );
                    
                    const size_t idx = getIdx(grid_x, grid_y);
                    if (frame_ev_[idx] == 0.f) 
                        touched_.push_back(idx);
                    if (evidence > frame_ev_[idx])
                        frame_ev_[idx] = evidence;
                }
            }
        }

        // Check if the sensor is reliable enough to mark misses
        const size_t health = frame_hp > 0 ? size_t(frame_hp) : pts.size();
        bool do_miss = health >= parameters_.min_points_for_miss;

        // Defining the sensor frustum
        const float frustum_xs[2] = {parameters_.roi_x_min, parameters_.roi_x_max};
        const float frustum_ys[2] = {-parameters_.roi_half_y_far, parameters_.roi_half_y_far};
        double min_x = 1e18;
        double max_x = -1e18;
        double min_y = 1e18;
        double max_y = -1e18;
        for (float frustum_x : frustum_xs) for (float frustum_y : frustum_ys) {
            const double warped_x = pose.x + cos_yaw * frustum_x - sin_yaw * frustum_y;
            const double warped_y = pose.y + sin_yaw * frustum_x + cos_yaw * frustum_y;
            min_x = std::min(min_x, warped_x);
            min_y = std::min(min_y, warped_y);
            max_x = std::max(max_x, warped_x);
            max_y = std::max(max_y, warped_y);
        }
        const int grid_x_0 = std::max(toGridNumber(min_x), origin_x_);
        const int grid_y_0 = std::max(toGridNumber(min_y), origin_y_);
        const int grid_x_1 = std::min(toGridNumber(max_x), width_ - 1);
        const int grid_y_1 = std::min(toGridNumber(max_y), height_ - 1);
        const float x_span = parameters_.roi_x_max - parameters_.roi_x_min;

        // ROI pass
        for (int grid_y = grid_y_0; grid_y <= grid_y_1; ++grid_y) {
            for (int grid_x = grid_x_0; grid_x <= grid_x_1; ++grid_x) {
                const double dx = (grid_x + 0.5) * parameters_.resolution - pose.x;
                const double dy = (grid_y + 0.5) * parameters_.resolution - pose.y;
                const float robot_x = float(cos_yaw * dx - sin_yaw * dy); // robot frame forward distance
                const float robot_y = float(-sin_yaw * dx + sin_yaw * dy); // robot frame lateral left distance
                if (robot_x < parameters_.roi_x_min || robot_x > parameters_.roi_x_max)
                    continue;
                
                const float t = (robot_x - parameters_.roi_x_min) / x_span; // how far forward the cell is with respect to the span
                const float half_width = parameters_.roi_half_y_near + 
                                         t * (parameters_.roi_half_y_far - parameters_.roi_half_y_near);
                if (std::fabs(robot_y) > half_width)
                    continue;

                const size_t idx = getIdx(grid_x, grid_y);
                const float evidence = frame_ev_[idx];
                if (evidence == 0.f && !do_miss)
                    continue;
                
                const float weight = 1.f / (
                    1.f + (robot_x / parameters_.range_ref) * (robot_x / parameters_.range_ref)
                );

                float delta = evidence * parameters_.l_hit;
                if (do_miss)
                    delta += (1.f - evidence) * parameters_.l_miss * weight;

                applyLogOdds(idx, delta);
                frame_ev_[idx] = 0.f;
            }
        }

        // Leftover splat cells which are just outside ROI get hit only
        for (size_t idx : touched_) {
            if (frame_ev_[idx] > 0.f) 
                applyLogOdds(idx, frame_ev_[idx] * parameters_.l_hit);
            frame_ev_[idx] = 0.f;
        }
        touched_.clear();
    }

    // --- QUERIES ---

    bool LaneMapper::isOccupied(double x, double y) const {
        const int grid_x = toGridNumber(x);
        const int grid_y = toGridNumber(y);
        return inWindow(grid_x, grid_y) && log_odds_[getIdx(grid_x, grid_y)] >= parameters_.l_occupied;
    }

    float LaneMapper::logOdds(double x, double y) const{
        const int grid_x = toGridNumber(x);
        const int grid_y = toGridNumber(y);
        return inWindow(grid_x, grid_y) ? log_odds_[getIdx(grid_x, grid_y)] : 0.f;
    }

    void LaneMapper::snapshot(std::vector<int8_t>& output, Window& window) const {
        output.assign(size_t(width_) * height_, -1);
        std::vector<uint8_t> occupied(size_t(width_) * height_, 0);

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const size_t idx = getIdx(origin_x_ + x, origin_y_ + y);
                const size_t out_idx = size_t(y) * width_ + x;
                if (!seen_[idx])
                    continue;
                output[out_idx] = 0;
                if (log_odds_[idx] >= parameters_.l_occupied)
                    occupied[out_idx] = 1; 
            }
        }

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (!occupied[size_t(y) * width_ + x])
                    continue;
                int count = 0;
                for (
                    int dy = std::max(0, y - parameters_.speckle_radius); 
                    dy < std::min(height_ - 1, y + parameters_.speckle_radius);
                    ++dy
                ) {
                    for (
                        int dx = std::max(0, x - parameters_.speckle_radius);
                        dx < std::min(width_ - 1, x + parameters_.speckle_radius);
                        ++dx
                    ) {
                        count += occupied[size_t(dy) * width_ + dx];
                    }
                }
                if (count >= parameters_.speckle_support)
                    output[size_t(y) * width_ + x] = 100;
            }
        }
        
        window = {
            origin_x_ * double(parameters_.resolution), 
            origin_y_ * double(parameters_.resolution),
            parameters_.resolution,
            width_, height_
        };
    }
}