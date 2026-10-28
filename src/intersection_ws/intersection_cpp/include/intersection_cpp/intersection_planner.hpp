#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nav_msgs/msg/occupancy_grid.hpp"

namespace intersection {
    enum class Turn : uint8_t {STRAIGHT = 0, LEFT = 1, RIGHT = 2};

    struct Vec2 {
        double x{0.0};
        double y{0.0};
    };
    
    struct Pose2D {
        double x{0.0};
        double y{0.0};
        double yaw{0.0};
    };

    struct Params {
        // Costmap params
        int occupied_threshold{100}; 

        // Road params
        double lane_center_ratio{0.25};
        double width_mismatch_tolerance{0.30}; // (|measured_width - road_width| / road_width) above which the left line is ignored

        // Lane line extraction params
        int min_line_pts{15}; // minimum inlier cells for line to be valid
        int ransac_iters{150};
        double max_heading_error{0.6}; // in radians (The line must be roughly parallel to bot's heading)
        double line_inlier_tol{0.12}; // Ransac inlier distance (in m)
        double roi_back{4.0};
        double roi_front{10.0};
        double fit_x_front{2.0}; // Only points upto this far ahead is used for line fit (in m)
        double max_line_gap{1.0}; // gap along the line that marks the intersection entrance (in m)
        double fallback_distance_to_inters{1.0}; // used if the line end cant be seen (in m)

        // Path generation params
        double min_turning_radius{1.5};
        double exit_length{2.0}; // Path length after intersection
        double merge_length{2.0}; // Distance used to blend the robot pose to lane center
        double path_resolution{0.1}; // Spaces between path poses (in m)

        double robot_radius{0.5}; // Used for clearance check (in m)
    };
   
    struct PlanDebug {
        bool used_right_line{false};
        bool used_left_line{false};
        bool intersection_visible{false};
        double measured_road_width{0.0};
        double dist_to_intersection{0.0};
        double lane_center_offset{0.0};
        double turn_radius{0.0};
        double inner_corner_clearance{0.0};
        double blocked_fraction{0.0};
    };

    struct PlanResult {
        bool success{false};
        std::string message;
        std::vector<Pose2D> path;
        PlanDebug debug;
    };
    
    class IntersectionPlanner {
        public:
            explicit IntersectionPlanner(const Params & params = Params()) : params_(params) {}

            void setParams(const Params & params) { params_ = params; }

            const Params & params() const { return params_; }

            PlanResult plan(
                const nav_msgs::msg::OccupancyGrid & map,
                const Pose2D & robot_pose,
                Turn turn, double road_width
            ) const;

        private:
            Params params_;
    };
}