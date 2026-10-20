cmake_minimum_required(VERSION 3.8)
project(lidar_cam_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)                 
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)    
find_package(geometry_msgs REQUIRED)        
find_package(sensor_msgs REQUIRED)
find_package(vision_interfaces REQUIRED)
find_package(fusion_interfaces REQUIRED)    
find_package(Eigen3 REQUIRED)               

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(fusion_node src/depth_estimator.cpp)
ament_target_dependencies(fusion_node
  rclcpp
  message_filters
  tf2
  tf2_ros
  tf2_sensor_msgs
  geometry_msgs
  sensor_msgs
  vision_interfaces
)

add_executable(in_lane_filter src/in_lane_filter.cpp)
ament_target_dependencies(in_lane_filter
  rclcpp
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  vision_interfaces
)

add_executable(obstacle_memory src/obstacle_memory.cpp)
ament_target_dependencies(obstacle_memory
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  vision_interfaces
  fusion_interfaces
)
target_link_libraries(obstacle_memory Eigen3::Eigen)

install(TARGETS fusion_node in_lane_filter obstacle_memory
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()