cmake_minimum_required(VERSION 3.8)
project(mapping_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(path_bridge src/lane_follow_test.cpp)

ament_target_dependencies(path_bridge
  rclcpp
  rclcpp_action
  nav_msgs
  nav2_msgs
)

# 1. Install the C++ executable target into lib/
install(TARGETS
  path_bridge
  DESTINATION lib/${PROJECT_NAME}
)

# 2. Install launch and config directories into share/
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()