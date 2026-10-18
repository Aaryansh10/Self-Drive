#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lane_mapper {
    // Points in robot frame
    struct Point2 {
        float x; // forward distance in m
        float y; // distance to the left in m
    };

    // Robot pose in odom frame
    struct Pose2 { 
        double x;
        double y;
        double yaw;
    };

    struct Params {
        // Map parameters
        float resolution = 0.1f;
        int width = 200;
        int height = 200;

        // Valid IPM footprint
        float roi_x_min = 0.3f;
        float roi_x_max = 2.5f;
        float roi_half_y_near = 0.5f;
        float roi_half_y_far = 1.2f;

        // Splat kernel: sigma = base + per_m * range
        float sigma_base = 0.09f;
        float sigma_per_m = 0.01f;
        float sigma_per_m2 = 0.0f;
        float range_ref = 2.0f;

        // Log-odds model parameters
        float l_hit = 0.9f;
        float l_miss = -0.12f;
        float l_min = -2.0f;
        float l_max = 5.0f;
        float l_occupied = 1.5f; // minimum threshold for a cell to be marked as occupied

        size_t min_points_for_miss = 15; // minimum number points needed in the frame to update the map

        // Output speckle filter (If a single pixel is marked then it checks its surroundings and updates accordingly)
        int speckle_radius = 1; // 3x3 window
        int speckle_support = 3; // minimum occupied cells in window
    };

    struct Window {
        // Geometry of the snapshot
        double origin_x; // world x-coordinate of the bottom left cell
        double origin_y; // world y-coordinate of the bottom left cell
        double resolution;
        int width;
        int height;
    };
    
    class LaneMapper {
        public:
            explicit LaneMapper(const Params& p = Params());

            void reset();
            // Resets the entire map

            void insert(const std::vector<Point2>& pts, const Pose2& pose, int frame_hp = -1);
            /*
                Takes the leading edge points of the white lane lines, and the pose at that snapshot
                and then splashes them onto the map using log-odds.

                frame_hp: The number of points used to determine if the frame is to be considered missed.
                It has the combined count of all the classes.
            */

            bool isOccupied(double x, double y) const;
            // Query for the position of a point in world coordinates

            float logOdds(double x, double y) const;
            // Query for the log odds of a point in world coordinates

            void snapshot(std::vector<int8_t>& output, Window& window) const;
            /*
                Unrolled, speckle filtered snapshot, Row-major, (0, 0) is bottom left
                -1 = unknown, 0 = free space, 100 = occupied.
            */

        private:
            int toGridNumber(double v) const; 
            // Converts distance in 'm' to grid cell numbers
            
            size_t getIdx(int grid_x, int grid_y) const; 
            // Converts 2D grid coordinate to 1D index

            bool inWindow(int grid_x, int grid_y) const;
            // Checks if the point is actually on the grid

            void applyLogOdds(size_t i, float d);
            // Does the actual log-odds maths

            // Functions to implement the rolling window
            void clearCol(int grid_x);
            // Clears the columns that fall off the grid as the bot moves

            void clearRow(int grid_y);
            // Clears the rows that fall off the grid as the bot moves

            void recenter(const Pose2& pose);
            // Checks if the robot has moved towards the edge of the map. If so then it shifts the entire map.

            // Parameters for the map
            Params parameters_;
            int width_, height_;
            int origin_x_ = 0, origin_y_ = 0;
            bool initialised_ = false;

            std::vector<float> log_odds_; // Stores the probability score for every cell
            std::vector<float> frame_ev_;
            std::vector<uint8_t> seen_;
            std::vector<size_t> touched_;
    };
}