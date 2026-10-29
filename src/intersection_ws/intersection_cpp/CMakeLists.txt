cmake_minimum_required(VERSION 3.8)
project(intersection_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find necessary dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED) # Assuming use of actions based on the workspace
find_package(intersection_interfaces_cpp REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(intersection_server 
  src/IntersectionServer.cpp
  src/intersection_planner.cpp
)

# 2. Add the include directory so the compiler can find intersection_planner.hpp
target_include_directories(intersection_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 3. Link the dependencies to your executable
ament_target_dependencies(intersection_server
  rclcpp
  rclcpp_action
  intersection_interfaces_cpp
  tf2_ros
)

# 4. Install the executable and include directory so ROS 2 can find them
install(TARGETS intersection_server
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
