# parking (slot detector v0.1)

Build:   colcon build --packages-select parking && source install/setup.bash
Run:     ros2 launch parking slot_detector.launch.py
RViz:    fixed frame `odom`; add MarkerArray /parking/markers and PointCloud2 /parking/map_points
Lock:    ros2 service call /parking/lock_best std_srvs/srv/Trigger   (unlock: /parking/unlock)
Tests:   python3 test/test_slot_core.py        (synthetic, no ROS needed)

Slot frame: origin = entrance centre, +x = into the slot. Green marker = U seen (back line found),
yellow = partial (two sides only, depth is a lower bound).
Tune first: width_range / depth_min in config/slot_detector.yaml (measure the real sim slots).
