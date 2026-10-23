cmake_minimum_required(VERSION 3.8)
project(bt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(fusion_interfaces REQUIRED)
find_package(lane_change_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(bt_executor src/bt_executor.cpp)
target_include_directories(bt_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(bt_executor rclcpp rclcpp_action std_msgs fusion_interfaces lane_change_interfaces)
target_link_libraries(bt_executor behaviortree_ros2::behaviortree_ros2)
 
install(TARGETS bt_executor DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch trees DESTINATION share/${PROJECT_NAME})
 

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
  add_executable(test_decision test/test_decision.cpp)
  target_include_directories(test_decision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME test_decision COMMAND test_decision)

  
endif()

ament_package()
