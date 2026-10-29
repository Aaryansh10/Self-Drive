#include "intersection_cpp/intersection_planner.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace intersection {
    namespace {
        constexpr double pi = 3.14159265358979323846;

        inline Vec2 operator+(Vec2 a, Vec2 b) {
            return {a.x + b.x, a.y + b.y};
        }

        inline Vec2 operator-(Vec2 a, Vec2 b) {
            return {a.x - b.x, a.y - b.y};
        }

        inline Vec2 operator*(Vec2 a, double k) {
            return {a.x * k, a.y * k};
        }

        inline double dot(Vec2 a, Vec2 b) {
            return a.x * b.x + a.y * b.y;
        }

        inline double norm(Vec2 a) {
            return std::hypot(a.x, a.y);
        }

        inline Vec2 normalized(Vec2 a) {
            const double n = norm(a);
            return n > 1e-9? Vec2{a.x / n, a.y / n} : Vec2{1.0, 0.0};
        }

        inline Vec2 leftNormal(Vec2 a) {
            return {-a.y, a.x};
        }

        class Grid {
            public:
                Grid(const nav_msgs::msg::OccupancyGrid & m, int occupied_threshold):
                     msg_(m), thresh_(occupied_threshold), res_(m.info.resolution),
                     width_(static_cast<int>(m.info.width)), height_(static_cast<int>(m.info.height)),
                     origin_x_(m.info.origin.position.x), origin_y_(m.info.origin.position.y)
                {
                    const auto & q = m.info.origin.orientation;
                    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
                    
                    cos_yaw_ = std::cos(yaw);
                    sin_yaw_ = std::sin(yaw);
                }

                int width() const {
                    return width_;
                }

                int height() const {
                    return height_;
                }

                bool occupied(int i, int j) const {
                    return static_cast<int>(msg_.data[static_cast<size_t>(j) * width_ + i]) >= thresh_;
                }

                Vec2 cellToWorld(int i, int j) const {
                    const double x = (i + 0.5) * res_;
                    const double y = (j + 0.5) * res_;
                    return {
                        origin_x_ + cos_yaw_* x - sin_yaw_ * y,
                        origin_y_ + sin_yaw_ * x + cos_yaw_ * y
                    };
                }

                bool worldToCell(Vec2 p, int & i, int & j) const {
                    const double dx = p.x - origin_x_;
                    const double dy = p.y - origin_y_;
                    const double x = cos_yaw_ * dx + sin_yaw_ * dy;
                    const double y = -sin_yaw_ * dx + cos_yaw_ * dy;
                    i = static_cast<int>(std::floor(x / res_));
                    j = static_cast<int>(std::floor(y / res_));
                    return i >= 0 && j >= 0 && i < width_ && j < height_;
                }

                bool occupiedWithin(Vec2 p, double r) const {
                    // Is true if the any occupied cell center lies within r of p

                    int cell_i, cell_j;
                    const int span = static_cast<int>(std::ceil(r / res_)) + 1; // p maybe sligthly outside the grid so we clamp the search window
                    worldToCell(p, cell_i, cell_j);

                    for (int j = std::max(0, cell_j - span); j <= std::min(height_ - 1, cell_j + span); ++j) {
                        for (int i = std::max(0, cell_i - span); i <= std::min(width_ - 1, cell_i + span); ++i) {
                            if (occupied(i, j) && norm(cellToWorld(i, j) - p) <= r)
                                return true;
                        }
                    }

                    return false;
                }

                double extentAlong(Vec2 p, Vec2 dir) const{
                    // Distance we can march from p along dir before leaving the grid

                    int i, j;
                    double t = 0.0;
                    while (t < 200.0 && worldToCell(p + dir * t, i, j)) {
                        t += res_;
                    }
                    return t;
                }

            private:
                const nav_msgs::msg::OccupancyGrid & msg_;
                int thresh_;
                double res_;
                int width_, height_;
                double origin_x_, origin_y_;
                double cos_yaw_, sin_yaw_;
        };

        struct Line {
            Vec2 centroid; // Centroid of the inliers
            Vec2 dir; // Unit direction
        };

        bool fitLine(const std::vector<Vec2> & pts, const Params & p, Line & out) {
            // RANSAC for a roughly forward line, refined with total least squares (PCA)

            if (static_cast<int>(pts.size()) < p.min_line_pts)
                return false;

            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);

            int best = 0;
            Vec2 best_pt{}, best_dir{1.0, 0.0};
            for (int it = 0; it < p.ransac_iters; ++it) {
                const size_t a = pick(rng); 
                const size_t b = pick(rng);
                if (a == b)
                    continue;

                Vec2 d = pts[a] - pts[b];
                const double len = norm(d);
                if (len < 0.5)
                    continue; // The points are too close and hence direction is unstable

                d = d * (1.0 / len);

                if (d.x < 0) 
                    d = d * -1.0;

                if (std::fabs(std::atan2(d.y, d.x)) > p.max_heading_error)
                    continue;

                const Vec2 n = leftNormal(d);
                int count = 0;
                for (const auto & pt: pts) {
                    if (std::fabs(dot(n, pt - pts[a])) <= p.line_inlier_tol)
                        count++;
                }
                
                if (count > best) {
                    best = count;
                    best_pt = pts[a];
                    best_dir = d;
                }
            }

            if (best < p.min_line_pts)
                return false;

            // Refine on inliers using 2 PCA passes
            for (int pass = 0; pass < 2 ; ++pass) {
                const Vec2 n = leftNormal(best_dir);
                Vec2 mean{};

                int count = 0;
                for (const auto & pt : pts) {
                    if (std::fabs(dot(n, pt - best_pt)) <= p.line_inlier_tol) {
                        mean = mean + pt;
                        ++count;
                    }
                }

                if (count < p.min_line_pts)
                    return false;

                mean = mean * (1.0 / count);

                double xx = 0, yy = 0, xy = 0;
                for (const auto & pt : pts) {
                    if (std::fabs(dot(n, pt - best_pt)) <= p.line_inlier_tol) {
                        const Vec2 r = pt - mean;
                        xx += r.x * r.x; yy += r.y * r.y; xy += r.x * r.y;
                    }
                }

                const double theta = 0.5 * std::atan2(2.0 * xy, xx - yy);
                best_dir = {std::cos(theta), std::sin(theta)};
                if (best_dir.x < 0)
                    best_dir = best_dir * -1.0;

                best_pt = mean;
            }

            out.centroid = best_pt;
            out.dir = best_dir;
            return true;
        }

        inline double smoothstep5(double t) {
            t = std::min(1.0, std::max(0.0, t));
            return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
        }

    }

    PlanResult IntersectionPlanner::plan(
        const nav_msgs::msg::OccupancyGrid & map, const Pose2D & robot_pose,
        Turn turn, double road_width
    ) const
    {
        PlanResult result;
        
        auto fail = [&result](const std::string & msg) {
            result.success = false;
            result.message = msg;
            result.path.clear();
            return result;
        };

        if (!(road_width > 0.0))
            return fail("Road width must be > 0");
        
        if (
            map.info.resolution <= 0.0f || map.info.width == 0 || map.info.height == 0 ||
            map.data.size() != static_cast<size_t>(map.info.width) * map.info.height
        ) 
            return fail("Local map is empty or is inconsistent");

        const double y_split = (1.0 - 2.0 * params_.lane_center_ratio) * road_width * 0.5;

        std::string warn;
        Grid grid(map, params_.occupied_threshold);

        const double cos_yaw = std::cos(robot_pose.yaw);
        const double sin_yaw = std::sin(robot_pose.yaw);

        std::vector<Vec2> all, right_cand, left_cand;
        for (int j = 0; j < grid.height(); ++j) {
            for (int i = 0; i < grid.width(); ++i) {
                if (!grid.occupied(i, j))
                    continue;

                const Vec2 world_coord = grid.cellToWorld(i, j);
                const double dx = world_coord.x - robot_pose.x;
                const double dy = world_coord.y - robot_pose.y;
                const Vec2 pt_robot_frame{cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy};

                if (pt_robot_frame.x < -params_.roi_back || pt_robot_frame.x > params_.roi_front)
                    continue;

                if (pt_robot_frame.y < y_split - 1.5 * road_width || pt_robot_frame.y > y_split + 1.5 * road_width)
                    continue;
                
                all.push_back(pt_robot_frame);
                
                if (pt_robot_frame.x <= params_.fit_x_front) {
                    if (pt_robot_frame.y < y_split && pt_robot_frame.y >= y_split - road_width)
                        right_cand.push_back(pt_robot_frame);
                    else if (pt_robot_frame.y >= y_split && pt_robot_frame.y < y_split + road_width)
                        left_cand.push_back(pt_robot_frame);
                }
            }
        }

        Line R{}, L{};
        bool hasRight = fitLine(right_cand, params_, R);
        bool hasLeft = fitLine(left_cand, params_, L);
        if (!hasRight && !hasLeft) 
            return fail("No lane line detected in local_map");
        
        Vec2 dir = hasRight && hasLeft ? normalized(R.dir + L.dir) : (hasRight ? R.dir : L.dir);
        Vec2 left_norm = leftNormal(dir);
        double latRight = hasRight ? dot(left_norm, R.centroid) : 0.0;
        double latLeft = hasLeft ? dot(left_norm, L.centroid) : 0.0;

        if (hasRight && hasLeft) {
            result.debug.measured_road_width = latLeft - latRight;
            if (std::fabs((latLeft - latRight) - road_width) > params_.width_mismatch_tolerance * road_width) {
                std::ostringstream ostr;
                ostr << "measured road width " << (latLeft - latRight) << " m differs from requested " <<
                        road_width << " m, ignoring left line; ";
                warn += ostr.str();
                hasLeft = false;
                dir = R.dir;
                left_norm = leftNormal(dir);
                latRight = dot(left_norm, R.centroid);
            }
        }

        double lane_center;
        if (hasRight && hasLeft)
            lane_center = 0.5 * ((latRight + params_.lane_center_ratio * road_width) + 
                        (latLeft - (1.0 - params_.lane_center_ratio) * road_width));
        else if (hasRight) 
            lane_center = latRight + params_.lane_center_ratio * road_width;
        else 
            lane_center = latLeft - (1.0 - params_.lane_center_ratio) * road_width;
        result.debug.used_right_line = hasRight;
        result.debug.used_left_line = hasLeft;
        result.debug.lane_center_offset = lane_center;

        const Vec2 dir_map{cos_yaw * dir.x - sin_yaw * dir.y, sin_yaw * dir.x + cos_yaw * dir.y};
        const double limit = std::min(params_.roi_front, grid.extentAlong({robot_pose.x, robot_pose.y}, dir_map));
        const double seed = -0.5;

        auto lineEnd = [&](double lat, double & end, bool & visible) -> bool {
            std::vector<double> ss;
            for (const auto & pt : all) {
                if (std::fabs(dot(left_norm, pt) - lat) <= 1.5 * params_.line_inlier_tol)
                    ss.push_back(dot(dir, pt));
            }
            
            if (ss.empty())
                return false;
            
            std::sort(ss.begin(), ss.end());
            bool gap = false;
            double seg_end = ss.back();
            for (size_t i = 1; i < ss.size(); ++i) {
                if (ss[i] - ss[i - 1] > params_.max_line_gap && ss[i - 1] >= seed) {
                seg_end = ss[i - 1];
                gap = true;
                break;
                }
            }

            end = seg_end;
            visible = gap || seg_end < limit - 0.5;
            return true;
        };

        double ends_sum = 0.0;
        int ends_n = 0;
        bool all_visible = true;
        for (int k = 0; k < 2; ++k) {
            if ((k == 0 && !hasRight) || (k == 1 && !hasLeft))
                continue;
            double e;
            bool vis;
            if (lineEnd(k == 0 ? latRight : latLeft, e, vis)) {
                ends_sum += e;
                ++ends_n;
                all_visible = all_visible && vis;
            }
        }
        
        double s_end;
        if (ends_n > 0 && all_visible) {
            s_end = ends_sum / ends_n;
            result.debug.intersection_visible = true;
        }
        else {
            s_end = params_.fallback_distance_to_inters;
            result.debug.intersection_visible = false;
            warn += "lane line end not visible, using fallback distance; ";
        }
        result.debug.dist_to_intersection = s_end;
        if (s_end < 0.0) 
            warn += "robot appears to be already inside the intersection; ";
        
        const double sigma = (turn == Turn::LEFT) ? 1.0 : -1.0;
        double Vs = 0.0, Rnat = 0.0, Rt = 0.0, T1 = 0.0, straight_end;
        
        if (turn == Turn::STRAIGHT)
            straight_end = s_end + road_width + params_.exit_length;
        else {
            Rnat = (turn == Turn::LEFT ? (1.0 - params_.lane_center_ratio) : params_.lane_center_ratio) * road_width;
            Vs = s_end + Rnat;
            if (Vs < 0.3)
                return fail("Intersection is behind or too close to plan a turn");
            Rt = std::max(Rnat, params_.min_turning_radius);
            T1 = Vs - Rt;
            
            if (T1 < 0.0) {
                std::ostringstream ostr;
                ostr << "too close to the intersection for min_turn_radius, radius reduced from " << Rt << " to " << Vs << " m; ";
                warn += ostr.str();
                Rt = Vs;
                T1 = 0.0;
            }

            straight_end = T1;
            result.debug.turn_radius = Rt;
            const Vec2 corner{s_end, sigma * Rnat}, centre{T1, sigma * Rt};
            result.debug.inner_corner_clearance = std::fabs(Rt - norm(corner - centre));
            if (result.debug.inner_corner_clearance < params_.robot_radius) 
                warn += "arc passes close to the inner lane-line corner; ";
        }

        const double step = std::max(0.02, params_.path_resolution);
        std::vector<Vec2> lane;
        const double l0 = -lane_center; // robot's lateral offset wrt the lane_center
        const double merge = std::min(params_.merge_length, straight_end);
        
        for (double s = 0.0; s < straight_end - 1e-6; s += step) {
            const double t = merge > 1e-6 ? s / merge : 1.0;
            lane.push_back({s, l0 * (1.0 - smoothstep5(t))});
        }

        if (turn == Turn::STRAIGHT)
            lane.push_back({straight_end, 0.0});
        else {
            const int na = std::max(4, static_cast<int>(std::ceil(Rt * pi / 2.0 /step)));
            for (int k = 0; k <= na; ++k) {
                const double th = 0.5 * pi * k / na;
                lane.push_back({T1 + Rt * std::sin(th), sigma * Rt * (1.0 - std::cos(th))});
            }
            for (double t = step; t <= params_.exit_length + 1e-6; t += step) {
                lane.push_back({Vs, sigma * (Rt + t)});
            }
        }

        // lane frame -> robot frame -> map frame
        std::vector<Vec2> pm;
        pm.reserve(lane.size());
        for (const auto & p : lane) {
            const Vec2 r = dir * p.x + left_norm * (lane_center + p.y); // robot frame
            pm.push_back({
                robot_pose.x + cos_yaw * r.x - sin_yaw * r.y, 
                robot_pose.y + sin_yaw * r.x + cos_yaw * r.y
            });
        }
        result.path.resize(pm.size());
        for (size_t i = 0; i < pm.size(); ++i) {
            const Vec2 seg = (i + 1 < pm.size()) ? pm[i + 1] - pm[i] : pm[i] - pm[i - 1];
            result.path[i] = {pm[i].x, pm[i].y, std::atan2(seg.y, seg.x)};
        }

        // Validation
        size_t blocked = 0;
        for (const auto & p : result.path) {
            if (grid.occupiedWithin({p.x, p.y}, params_.robot_radius))
                ++blocked;
        }
        result.debug.blocked_fraction = static_cast<double>(blocked) / result.path.size();
        if (blocked > 0) {
            std::ostringstream ostr;
            ostr << static_cast<int>(100.0 * result.debug.blocked_fraction) 
                << "% of path poses are within robot_radius of an occupied cell; ";
            warn += ostr.str();
        }
        result.success = true;
        result.message = warn.empty() ? "OK" : "OK (" + warn + ")";
        return result;
    }
}