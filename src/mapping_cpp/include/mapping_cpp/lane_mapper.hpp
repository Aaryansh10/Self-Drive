#pragma once
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
        float resolution = 0.1f;
        int width = 200;
        int height = 200;


    };
    
    
}