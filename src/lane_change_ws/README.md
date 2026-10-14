LANE CHANGE :

T1

ros2 topic hz /odom
ros2 launch lane_change lane_change.launch.py

T2

root@sunanda-OMEN:/workspace/Self-Drive# ros2 topic pub -r 10 /centerline/lane_1 nav_msgs/msg/Path "{header: {frame_id: link_base}, poses: [{pose: {position: {x: 1.5, y: 0.0}}}, {pose: {position: {x: 3.0, y: 0.0}}}, {pose: {position: {x: 4.5, y: 0.0}}}, {pose: {position: {x: 6.0, y: 0.0}}}, {pose: {position: {x: 7.5, y: 0.0}}}]}" > /dev/null &
[1] 621

root@sunanda-OMEN:/workspace/Self-Drive# ros2 topic pub -r 10 /centerline/lane_2 nav_msgs/msg/Path "{header: {frame_id: link_base}, poses: [{pose: {position: {x: 1.5, y: 3.048}}}, {pose: {position: {x: 3.0, y: 3.048}}}, {pose: {position: {x: 4.5, y: 3.048}}}, {pose: {position: {x: 6.0, y: 3.048}}}, {pose: {position: {x: 7.5, y: 3.048}}}]}" > /dev/null &
[2] 644

root@sunanda-OMEN:/workspace/Self-Drive# ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.5}}"

T3

ros2 action send_goal /lane_change lane_change_interfaces/action/LaneChange \
"{obstacle_distance: 25.0, completion_distance: 0.0, target_speed: 0.0, target_lane_free: true}" \
--feedback


T4
ros2 topic echo /lane_change/active
ros2 topic hz /lane_change/path


check the yaml file once and run along with center line to check for obstacle distance , check sever timimgs of lane change acttion 