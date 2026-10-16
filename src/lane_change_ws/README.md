lane_change_ws
  lane_change_interfaces : LaneChange.action (goal / feedback / result of the lane change)
  lane_change            : lane_change_server (the action), lane_path_mux (picks the one path for the follower),
                           planner core, two small offline checks
Run: ros2 launch lane_change lane_system.launch.py   (needs the two centreline topics and /odom)
The follower subscribes to /follow_path (nav_msgs/Path, odom frame). Nothing here depends on Nav2.
