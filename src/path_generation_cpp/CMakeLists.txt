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
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ROS-free algorithm library
add_library(centerline_generator STATIC src/centerline_generator.cpp)
target_include_directories(centerline_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# ROS node
add_executable(centerline_node src/centerline_node.cpp)
target_link_libraries(centerline_node centerline_generator)
ament_target_dependencies(centerline_node rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs)

# Standalone replay/benchmark tool (no ROS)
add_executable(centerline_bench test/bench_centerline.cpp)
target_link_libraries(centerline_bench centerline_generator)

install(TARGETS centerline_node centerline_bench DESTINATION lib/${PROJECT_NAME})

ament_package()