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

The centerline node pairs skeleton maps by timestamp, transforms cell centers into `base_frame` using acquisition-time TF, crops to the forward planning region, and traces ordered branches through adjacent cells. Explicit segment IDs prevent disconnected lanes or junction branches being joined merely because their endpoints are close. Path topics remain `nav_msgs/Path` in `link_base`. Persistence lives in the maps; previous robot-frame centerlines are not blended across robot motion. Empty paired maps or TF/geometry errors publish empty paths.

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

The mapper retains its ROI, Gaussian splat, log-odds, speckle, and TF wait parameters. The old mapper `map_topic` parameter is replaced by `white_map_topic`. Centerline `pc_whitelane_topic` / `pc_yellowlane_topic` are replaced by the skeleton-topic parameters above. Spatial centerline smoothing and road-width parameters remain available; temporal memory is reset for each map pair.

## Build and run

```bash
colcon build --packages-select vision_cpp path_generation_cpp mapping_cpp --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
ros2 launch steve_description stack.launch.py
```

`stack.launch.py` starts the full pipeline. For a standalone mapping/navigation setup, `ros2 launch mapping_cpp mapping.launch.py` starts mapping, skeletonization, centerlines, and the existing Nav2 nodes; supply vision/clouds and TF separately. Use one launch entrypoint at a time to avoid duplicate map and path publishers.

## Checks

```bash
ctest --test-dir build/mapping_cpp -R '^lane_maps$' --output-on-failure
ctest --test-dir build/path_generation_cpp -R '^skeleton_map$' --output-on-failure
ROS_DOMAIN_ID=104 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/tmp/lane_ros_logs \
  /usr/bin/python3 src/mapping_cpp/test/test_map_pipeline.py
```

The ROS smoke test requires `rclpy`, `sensor_msgs_py`, `tf2_ros`, and a sourced install. It starts only the mapper, skeletonizer, and centerline node in the caller's ROS domain, injects synthetic cleaned lane clouds with translated/rotated TF, checks separate maps and both paths, then verifies empty-map clearing. It stops its own processes on exit and writes node logs under `/tmp`.
