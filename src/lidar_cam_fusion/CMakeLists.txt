cmake_minimum_required(VERSION 3.8)
project(lidar_cam_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_interfaces REQUIRED)
find_package(fusion_interfaces REQUIRED)

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
  tf2_ros 
  tf2_sensor_msgs 
  sensor_msgs 
  vision_interfaces 
  fusion_interfaces
)

install(TARGETS fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
