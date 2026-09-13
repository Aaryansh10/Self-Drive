# Self-Drive (Documentation)

## How to run *sim*:
* `cd Self-Drive`
* `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release`
* `source install/setup.bash`
* `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch self_drive_course self_drive_course.launch.py`
* `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch steve_description robot_gazebo.launch.py`
* `ros2 launch ros2 launch mapping_cpp mapping.launch.py `
