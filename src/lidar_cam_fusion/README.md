# `depth_estimator.cpp` 

## 1. Overview & Purpose
The **`depth_estimator`** node (implemented via the `Fusion` C++ class) is a ROS 2 perception fusion node. Its job is to take **2D bounding box detections** from a camera (e.g., cars, potholes, tyres, stop signs) and calculate **how far away** each detected object is in meters.


1. **Stop-Line Point Cloud Lookup (for Stop Signs):** Uses a dedicated point cloud representing the painted road stop line rather than the sign pole itself.
2. **Inverse Perspective Mapping / Ray-Ground Intersection (for Flat/Ground Objects):** Mathematically projects a ray from the camera pixel onto the flat road surface (used for potholes or tyres that LiDAR beams might miss or pass over).
3. **3D LiDAR Frustum Projection (for Standard Obstacles):** Projects 3D LiDAR points onto the 2D camera image, collects all points landing inside the 2D bounding box, and takes the 25th percentile depth.


## 2. Node Architecture: Topics & Interfaces

### Subscribed Topics
| Topic Name | Message Type | QoS Profile | Purpose |

| `zed_node/monocamera/camera_info` | `sensor_msgs/msg/CameraInfo` | `SensorDataQoS` | Provides the camera's intrinsic calibration matrix ($K$) and TF frame name. **Nothing runs until this arrives.** |
| `detections` | `vision_interfaces/msg/DetectionArray` | `SensorDataQoS` | 2D bounding boxes (`x1, y1, x2, y2`), class names, and confidence scores from the vision model. Synchronized with `obstacle_cloud`. |
| `obstacle_cloud` | `sensor_msgs/msg/PointCloud2` | `SensorDataQoS` | 3D obstacle point cloud (typically from LiDAR). Synchronized with `detections`. |
| `pc_stopline` | `sensor_msgs/msg/PointCloud2` | `SensorDataQoS` | Asynchronous point cloud containing points belonging to a detected stop line on the road. |

### Published Topics
| Topic Name | Message Type | QoS Queue | Purpose |

| `detected_distance` | `fusion_interfaces/msg/DetectionDistanceArray` | `10` | Publishes each input detection enriched with its computed `distance` (meters), `n_pts` (number of points used), and a `valid` flag. |

### TF2 Transforms

1. **`cloud->header.frame_id` $\rightarrow$ `camera_frame`:** Transforms 3D points from the LiDAR/stop-line sensor frame into the camera frame.
2. **`camera_frame` $\rightarrow$ `ground_frame`:** Finds where the camera is located and how it is tilted relative to the vehicle's ground/base frame (`link_base`) for ray-casting onto the road.

---

## 3. Core Logic: How the 3 Distance Methods Work

### Step 0: Synchronization (`message_filters`)
Camera images and LiDAR scans don't arrive at the exact same nanosecond. The node uses an **`ApproximateTime` synchronizer** (`sync`) with a queue size of `10` and a maximum allowed time difference of **0.1 seconds (100 ms)**. When a matching pair of `detections` and `obstacle_cloud` messages arrives within 100 ms of each other, `fusion_callback()` is triggered.

---

### Method A: Stop Sign Distance (Stop-Line Substitution)
When the autonomous vehicle sees a stop sign (`class_name == stop_sign_class`), the controller usually cares about **where to stop the car** (the painted line on the road), or uses a pre-filtered stop-line point cloud (`pc_stopline`).
1. **Asynchronous Caching (`stop_line_callback`):** Whenever a message arrives on `pc_stopline`, it is transformed into `camera_frame`.
2. **Filtering:** It loops through every point and keeps forward distances (`x > 0.1m`) that are finite numbers.
3. **Percentile Selection:** If at least `stop_line_min_points` (default: `5`) exist, it sorts them using `std::nth_element` and picks the **10th percentile** (`stop_line_percentile = 0.1`) to get the closest edge of the line while ignoring stray noise points. This distance is saved in `stop_line_dist`, and the timestamp is saved in `stop_line_stamp`.
4. **Lookup in `fusion_callback`:** When a stop sign is detected in the camera, the node checks:
   * Is `stop_line_valid == true`?
   * Is the saved stop-line measurement fresh? ($\text{age} \le \text{stop\_line\_max\_age}$, default `0.5s`).
   * If both pass, it assigns `stop_line_dist` to the stop sign detection.

---

### Method B: Inverse Perspective Mapping / IPM (Potholes & Tyres)
Objects listed in `ipm_classes` (default: `"pothole"`, `"tyre"`) are flat or small on the road, meaning sparse LiDAR rings might completely miss them. Instead of using LiDAR points, the node uses **geometry**:
1. **Find Box Center:** Computes the center pixel $(u, v)$ of the 2D bounding box:
   $$u = 0.5 \times (x_1 + x_2), \quad v = 0.5 \times (y_1 + y_2)$$
2. **Create a 3D Camera Ray (`ipm_depth`):** Converts pixel $(u, v)$ into a 3D ray vector pointing out of the camera lens, using camera intrinsics (`fx, fy, cx, cy`), and rotates it into the vehicle's `ground_frame` using rotation matrix `R`:
   $$\vec{d} = R \cdot \begin{bmatrix} 1.0 \\ -\frac{u - c_x}{f_x} \\ -\frac{v - c_y}{f_y} \end{bmatrix}$$
3. **Check Ray Direction:** If $d_z > -10^{-3}$, the ray is pointing parallel to the ground or up toward the sky—it will never hit the road, so it returns `false`.
4. **Intersect with the Ground Plane:** Calculates the distance scaling factor $t$ where the ray intersects the horizontal plane $z = \text{ground\_z}$:
   $$t = \frac{\text{ground\_z} - \text{origin}_z}{d_z}$$
5. If $t$ is positive and within `max_ipm_range` (default `40.0m`), $t$ is used as the object's distance.

---

### Method C: LiDAR Point Cloud Projection (All Other Classes)
For standard 3D obstacles (cars, pedestrians, cones, etc.), the node fuses the 3D `obstacle_cloud` with the 2D bounding box:
1. **Transform Cloud:** Transforms `obstacle_cloud` into `camera_frame`.
2. **Coordinate System Remapping:** Standard ROS link frames use **X-forward, Y-left, Z-up**, whereas standard optical pinhole math uses **Z-forward, X-right, Y-down**. Inside the loop, the code remaps each 3D point:
   * `x = -(*iter_y)` (Right)
   * `y = -(*iter_z)` (Down)
   * `z = *iter_x` (Forward depth)
3. **Pinhole Camera Projection:** Projects every valid 3D point ($z > 0.1\text{m}$) onto 2D image pixel coordinates $(u, v)$:
   $$u = \frac{f_x \cdot x}{z} + c_x, \quad v = \frac{f_y \cdot y}{z} + c_y$$
4. **Bounding Box Filter:** Checks if pixel $(u, v)$ falls inside the 2D detection box (`x1 <= u <= x2` and `y1 <= v <= y2`). If it does, the forward depth $z$ is added to the `depths` list.
5. **25th Percentile Distance:** Bounding boxes are rectangular, so some background LiDAR points (walls, trees behind the object) often sneak into the corners of the box. Taking the **25th percentile** (`depths.size() * 0.25`) instead of the average or minimum gives the front surface of the obstacle while rejecting isolated noise points in front of it. Requires at least **3 points** (`depths.size() >= 3`) to be marked `valid`.

---

## 4. Dictionary of Parameters & Class Member Variables

### ROS 2 Parameters (Configurable via Launch/YAML)
| Parameter | Type | Default | Description |

| `ipm_classes` | `string[]` | `["pothole", "tyre"]` | Object classes that use Ray-Ground Intersection (IPM) instead of LiDAR points. |
| `stop_sign_class` | `string` | `"sign"` | Class name that triggers the stop-line distance substitution logic. |
| `stop_line_max_age` | `double` | `0.5` | Maximum age (seconds) a cached stop-line distance is considered valid. |
| `stop_line_min_points`| `int` | `5` | Minimum number of valid points in `pc_stopline` required to compute stop-line distance. |
| `stop_line_percentile`| `double` | `0.1` | Percentile (0.0 to 1.0) of sorted stop-line point depths to use (`0.1` = 10th percentile). |
| `ground_frame` | `string` | `"link_base"` | The TF frame representing the vehicle base/ground reference for IPM. |
| `ground_z` | `double` | `0.0` | The Z-height (in meters) of the road surface inside `ground_frame`. |
| `max_ipm_range` | `double` | `40.0` | Maximum allowed distance (meters) for an IPM calculation before it is rejected as invalid. |

### Internal State Variables (Private Members)
* **`fx, fy, cx, cy` (`double`):** Focal lengths and optical centers extracted from the camera's $3 \times 3$ intrinsic matrix $K$.
* **`ok` (`bool`):** Gatekeeper flag. Starts `false`, becomes `true` once the first `CameraInfo` message is received.
* **`camera_frame` (`std::string`):** Stores the TF frame ID of the camera (taken from `CameraInfo.header.frame_id`).
* **`stop_line_valid` (`bool`):** Whether the latest `pc_stopline` cloud had enough points (`>= stop_line_min_points`).
* **`stop_line_dist` (`float`):** The latest calculated distance to the stop line.
* **`stop_line_n` (`size_t`):** Total valid points found in the latest `pc_stopline` cloud.
* **`stop_line_stamp` (`rclcpp::Time`):** Timestamp of when the latest `pc_stopline` cloud was captured.





# `obstacle_memory.cpp`

