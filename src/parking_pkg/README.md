# PARKING

-> pull in
-> pull out
-> parallel parking

Basically , instead of detecting parking spaces independently from individual frames, accumulate observed white-line points in a common odometry coordinate frame, extract geometric line segments, identify parking-space configurations from those segments, and maintain persistent slot identities over time


   White-Line Point Cloud
            |
 Coordinate Transformation   link_base → odom
            |
    Range Filtering  0.8m - 10m
            |
          ACCUMULATED POINT MAP
                VoxelMap
                   |
                   v
             Map Pruning
                   |
                   v
           Point Extraction
          (Minimum voxel hits)
                   |
                   v
          SEQUENTIAL RANSAC
           Line Extraction
                   |
                   v
          PCA Line Refinement
                   |
                   v
          Segment Gap Splitting
                   |
                   v
          Junction Identification
              & Splitting
                   |
                   v
          PARKING SLOT DETECTION
                   |
           +-------+-------+
           |               |
           v               v
      Complete Slot   Partial Slot
       U-shaped       Two parallel
       3 boundaries   side boundaries
           |               |
           +-------+-------+
                   |
                   v
             Slot Deduplication
                   |
                   v
             SLOT TRACKER
                   |
          +--------+--------+
          |        |        |
          v        v        v
       Matching  Smoothing  Confidence
          |        |        |
          +--------+--------+
                   |
                   v
           TRACKED PARKING SLOTS
                   |
          +--------+---------+
          |        |         |
          v        v         v
       PoseArray  SlotInfo  RViz Markers