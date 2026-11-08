cmake_minimum_required(VERSION 3.8)
project(path_generation_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # the whole point of this package is speed
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ROS-free algorithm library
add_library(centerline_generator STATIC src/centerline_generator.cpp src/road_tracker.cpp)
target_include_directories(centerline_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# ROS node
add_executable(centerline_node src/centerline_node.cpp)
target_link_libraries(centerline_node centerline_generator)
ament_target_dependencies(centerline_node rclcpp message_filters tf2_ros nav_msgs geometry_msgs std_msgs)

# Standalone replay/benchmark tool (no ROS)
add_executable(centerline_bench test/bench_centerline.cpp)
target_link_libraries(centerline_bench centerline_generator)

install(TARGETS centerline_node centerline_bench DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_skeleton_map test/test_skeleton_map.cpp)
  target_link_libraries(test_skeleton_map centerline_generator)
  add_test(NAME skeleton_map COMMAND test_skeleton_map)
  add_executable(test_road_tracker test/test_road_tracker.cpp)
  target_link_libraries(test_road_tracker centerline_generator)
  add_test(NAME road_tracker COMMAND test_road_tracker)
endif()

ament_package()