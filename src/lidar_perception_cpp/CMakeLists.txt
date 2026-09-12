cmake_minimum_required(VERSION 3.8)
project(lidar_perception_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(PCL REQUIRED COMPONENTS common filters)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Build LidarProcessor as a shared library so it can be loaded into a
# component_container / component_container_mt (needed for intra-process
# comms with controller_server). rclcpp_components_register_node below
# also generates a standalone "lidar_processor" executable that loads
# this component, so `ros2 run lidar_perception_cpp lidar_processor`
# still works exactly as before — nothing changes on that front.
add_library(lidar_processor_component SHARED src/lidar_processor.cpp)
ament_target_dependencies(lidar_processor_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  pcl_conversions
  tf2
)
target_link_libraries(lidar_processor_component ${PCL_LIBRARIES})

rclcpp_components_register_node(lidar_processor_component
  PLUGIN "LidarProcessor"
  EXECUTABLE lidar_processor
)

install(TARGETS
  lidar_processor_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()