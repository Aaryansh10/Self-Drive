cmake_minimum_required(VERSION 3.8)
project(stop_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find all required ROS 2 packages based on your #includes
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(stop_interfaces REQUIRED)
find_package(vision_interfaces REQUIRED)

# 2. Declare the executable target
add_executable(stop_action src/stop_action.cpp)

# 3. Add the include directory so "stop_action/stop_logic.hpp" resolves
target_include_directories(stop_action PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 4. Link dependencies to the executable
ament_target_dependencies(stop_action
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  stop_interfaces
  fusion_interfaces
)

# 5. Install the executable to lib/stop_action
install(TARGETS
  stop_action
  DESTINATION lib/${PROJECT_NAME}
)

# 6. Install the header files
install(DIRECTORY include/
  DESTINATION include
)

# 7. Install launch and config directories to share/stop_action
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()