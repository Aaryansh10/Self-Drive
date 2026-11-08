# Stop action integration

`ros2 launch bt bt.launch.py` starts the BT and stop server together. Do not also
start a second stop server. Perception, lane change, path bridge, and navigation
remain separate launches. All nodes must use the same `use_sim_time` setting.

The command chain in `mapping_cpp/launch/mapping.launch.py` is now:

```
controller -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel_drive
           -> stop_action_server -> /cmd_vel -> robot
```

The stop server must be running for navigation commands to reach the robot.
Other navigation launches must use the same remappings. No other publisher
should command `/cmd_vel` directly. The server forwards navigation during
MONITORING and overrides it during BRAKING/STOPPED. Approach timeout latches
zero commands until a new goal is accepted. Cancellation sends zero and releases
command ownership back to navigation.

## Behavior

- `pedestrian`: brake using the goal distance, then remain STOPPED while any
  pedestrian is in lane. Fresh `/detections_lane` observations must show the lane
  clear continuously for 0.5 seconds before success. Invalid pedestrian depth
  and stale/missing detection streams cannot release the stop. This uses the
  upstream `is_in_lane` classification, including its configured unknown-lane
  policy. There is no 30-second timeout while STOPPED.
- `stop_sign`: brake at the corresponding stop line, remain stationary for two seconds, then
  succeed. A pedestrian in lane or stale perception extends the hold. The BT
  remembers the completed obstacle so it can proceed to its next decision.
- Barrels, tyres and potholes retain their lane-change behavior. The stop server
  rejects these classes. Pedestrians are never entered in completed-stop memory.
  The BT also reads live lane detections to avoid retriggering a cleared pedestrian
  from an old memory track; re-entry into the lane can trigger another stop.
- The XML Sequence keeps an accepted stop running even if its initiating
  detection disappears. Stop failures retry without falling through to a lane
  change. Lane following stays enabled to supply approach commands; the server
  owns braking and standstill commands.

The BT sends measured obstacle distance in `Stop.Goal`. Distances are propagated
using odometry during approach; pedestrian motion is not tracked by the braking
model. Fresh odometry is required to finish a stop. `hold_time` remains in the
wire interface for compatibility, but server policy determines the actual hold.

The current model's `sign` class denotes STOP signs. Fusion normalizes it to
`stop_sign`, uses the existing fresh `/pc_stopline` estimate, and preserves the
actual line point for obstacle memory; the BT uses its forward distance
(rather than projecting the distance along
the roadside sign's image ray). Obstacle memory now retains these detections.
A sign without a valid fresh line estimate does not yet produce a stop target.
There is no new sign-to-line association algorithm: the existing stop-line cloud
must represent the relevant line. If the model is changed to classify generic
traffic signs, its stop-sign classification must replace this class mapping.
The sign and its line form one stop event with one `stop_sign` goal and one
handled-memory entry. The line supplies geometry only; it is not a separate
BT class, action target, or braking profile.

## Build and checks

Rebuild `stop_interfaces` along with its clients: an older generated interface
in this workspace lacked `obstacle_distance`. Required BT dependencies are
BehaviorTree.CPP v4 (`behaviortree_cpp`) and `behaviortree_ros2`; Humble's
`behaviortree_cpp_v3` is not a substitute.

```
colcon build --packages-select stop_interfaces stop_action bt lidar_cam_fusion mapping_cpp
source install/setup.bash
```

Use a fresh build/install base if an existing build mixes symlink and regular
Python interface installations.

Decision regression tests require no ROS runtime:

```
g++ -std=c++17 -Isrc/bt/include src/bt/test/test_decision.cpp -o /tmp/test_decision
/tmp/test_decision
```

The live server regression check starts its own server in ROS domain 87 and
publishes synthetic odometry/detections; run after sourcing the built workspace:

```
python3 src/stop_action_ws/stop_action/test/check_stop_server.py
```

It checks command override, rejected classes, indefinite pedestrian occupancy,
stale perception, release on clearance, two-second stop-sign dwell, and cancel.
Simulator testing is still needed to tune braking profiles against actual vehicle
dynamics and confirm upstream lane/sign classification.
