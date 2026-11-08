# Lane map pipeline

```text
vision_node (CUDA mask cleaning + IPM)
  /pc_whitelane ─┐
  /pc_yellowlane ┴─> mapping_node
                       /lane_map/white ──> map_skeleton_node ──> /lane_map/white_skeleton ─┐
                       /lane_map/yellow ─> map_skeleton_node ──> /lane_map/yellow_skeleton ┴─> centerline_node
                                                                                             /path/left_centerline
                                                                                             /path/right_centerline
```

Lane maps and skeleton maps are `nav_msgs/OccupancyGrid`: `-1` unknown, `0` background, `100` lane. All four map topics use reliable, transient-local QoS. White and yellow maps maintain independent log-odds evidence in matching rolling windows. The default window is 25 × 25 m at 0.1 m resolution; this is a local accumulated map, not an unbounded global map.

The mapper pairs white/yellow clouds by exact acquisition timestamp. Both must use the configured `base_frame` (`link_base` by default). It waits for the acquisition-time transform into `odom_frame` (`odom`) before insertion. Stop-line messages are no longer required to update lane maps. Empty lane clouds are valid; misses are applied when the combined valid lane-point count meets `min_points_for_miss`.

The skeleton node applies Zhang-Suen thinning to convergence, then removes small connected components. It preserves each input map's timestamp, frame, resolution, dimensions, origin, and unknown cells. It runs on the CPU independently of the CUDA vision process.

The centerline node pairs skeleton maps by timestamp, transforms cell centers into `base_frame` using the latest available TF pose, and traces ordered branches. A road tracker associates branches with the current road's outer boundaries and yellow divider, using their distance and direction relative to a world-frame road reference. When several branches compete at the same distance, the closest match wins. Other roads remain in the maps but do not contribute to the paths. Compatible short fragments are joined before centerline generation. If the tracked road or selected lane no longer fits, the node can reacquire from multiple mutually consistent markings. The tracker uses the selected lane to initialize or reacquire the road and follows the `/lane_change/selected_lane` topic (1 = right, 2 = left). `initial_lane` is used until that topic arrives and defaults to `left`, matching `lane_change_server`.

The tracker stores the last generated paths in the map frame. During a brief loss of lane markings it can transform the remaining observed path into the current robot frame; it does not extrapolate beyond that path or refresh the fallback's age. By default, fallback expires after 2 seconds, 3 m of robot travel, a heading change of 0.5 rad, or if the remaining path starts more than 1 m away. Once these limits are exceeded, empty paths are published until the road is reacquired. Path timestamps correspond to the pose used for generation. Maps whose timestamp differs from the latest dynamic TF by more than `max_map_tf_skew` produce empty paths; static transforms do not impose that limit. A large heading change resets the road reference to allow reacquisition after a turn. TF and invalid-map errors also publish empty paths. The intersection behavior tree continues to use its separate intersection planner while lane following is disabled.

## Configuration

| Node | Parameters | Defaults |
|---|---|---|
| mapper | `white_topic`, `yellow_topic` | `/pc_whitelane`, `/pc_yellowlane` |
| mapper | `white_map_topic`, `yellow_map_topic` | `/lane_map/white`, `/lane_map/yellow` |
| mapper | `resolution`, `width`, `height`, `publish_every` | `0.1`, `250`, `250`, `3` |
| skeleton | `white_map_topic`, `yellow_map_topic` | same map topics |
| skeleton / centerline | `white_skeleton_topic`, `yellow_skeleton_topic` | `/lane_map/white_skeleton`, `/lane_map/yellow_skeleton` |
| skeleton | `occupied_threshold`, `min_component_cells` | `50`, `3` |
| centerline | `base_frame`, `max_range`, `max_lateral_range` | `link_base`, `8.0`, `12.0` |
| centerline | `max_map_tf_skew` | `1.0` s |
| centerline | `selected_lane_topic`, `initial_lane` | `/lane_change/selected_lane`, `left` |
| centerline | `road_boundary_tolerance`, `road_tolerance_per_m` | `2.0` m, `0.10` m/m |
| centerline | `road_min_forward_span`, `road_min_forward_alignment`, `road_max_branch_heading_error` | `0.25` m, `0.55`, `0.7` rad |
| centerline | `road_reference_timeout`, `road_reference_max_distance`, `road_reference_max_heading_change` | `8.0` s, `12.0` m, `0.9` rad |
| centerline | `path_fallback_timeout`, `path_fallback_max_distance`, `path_fallback_max_heading_change`, `path_fallback_max_anchor_distance` | `2.0` s, `3.0` m, `0.5` rad, `1.0` m |

Intersection decisions in `bt` require a recent `/pc_stopline` cloud containing a transverse line: by default at least 10 points, spanning 0.6 m, with a bounded slope and gaps. Sign-provided stop-line depth must also agree with the validated cloud. These thresholds are configured in `src/bt/config/bt.yaml`; sign detections alone cannot trigger an intersection.

The mapper retains its ROI, Gaussian splat, log-odds, speckle, and TF wait parameters. The old mapper `map_topic` parameter is replaced by `white_map_topic`. Centerline `pc_whitelane_topic` / `pc_yellowlane_topic` are replaced by the skeleton-topic parameters above. Spatial centerline smoothing and road-width parameters remain available. The legacy robot-frame generator memory is reset for each map pair; the road reference and fallback paths are tracked separately in the map frame. The association assumes a two-lane road with outer white boundaries and a yellow divider, using `nominal_road_width` (default 7.4 m). Tune the width and tolerances for the actual course.

## Build and run

```bash
colcon build --packages-select vision_cpp path_generation_cpp mapping_cpp --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
# Terminal 1: start Gazebo and the course
ros2 launch self_drive_course self_drive_course.launch.py
# Terminal 2: spawn the robot and publish its state/odom TF
ros2 launch steve_description robot_gazebo.launch.py
# Terminal 3: perception, planning, behavior tree and controller
ros2 launch steve_description stack.launch.py
```

`stack.launch.py` starts the perception and control pipeline; it does not start Gazebo or spawn the robot. The controller publishes `/cmd_vel_nav`, and the velocity smoother publishes the final `/cmd_vel` consumed by the simulated robot. The behavior tree node is named `bt` to match `bt.yaml`.

If the robot remains stationary, check `/clock`, the camera and `/pc_whitelane`/`/pc_yellowlane`, `/lane_map/white_skeleton`, `/path/left_centerline`, `/path/right_centerline`, `/path/selected_centerline`, `/lane_follow/enable`, the `/follow_path` action, `/cmd_vel_nav` and `/cmd_vel`. A published path with `poses: []` does not provide a goal. The selected lane comes from `/lane_change/selected_lane` (`1` right, `2` left). Controller logs reporting collision ahead mean that Nav2 is refusing to drive along the supplied path; inspect the local costmap and obstacles before changing collision settings.

`stack.launch.py` starts the full pipeline. For a standalone mapping/navigation setup, `ros2 launch mapping_cpp mapping.launch.py` starts mapping, skeletonization, centerlines, and the existing Nav2 nodes; supply vision/clouds and TF separately. Use one launch entrypoint at a time to avoid duplicate map and path publishers.

## Checks

```bash
ctest --test-dir build/mapping_cpp -R '^lane_maps$' --output-on-failure
ctest --test-dir build/path_generation_cpp --output-on-failure
ROS_DOMAIN_ID=104 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/tmp/lane_ros_logs \
  /usr/bin/python3 src/mapping_cpp/test/test_map_pipeline.py
ROS_DOMAIN_ID=104 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/tmp/lane_ros_logs \
  /usr/bin/python3 src/path_generation_cpp/test/test_centerline_tf_startup.py
```

The ROS smoke test requires `rclpy`, `sensor_msgs_py`, `tf2_ros`, and a sourced install. It starts only the mapper, skeletonizer, and centerline node in the caller's ROS domain, injects synthetic cleaned lane clouds with translated/rotated TF, checks separate maps and both paths, then verifies short-gap recovery and eventual empty-path expiry. It stops its own processes on exit and writes node logs under `/tmp`.
