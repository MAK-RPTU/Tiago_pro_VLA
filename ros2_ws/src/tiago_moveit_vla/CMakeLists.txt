cmake_minimum_required(VERSION 3.8)
project(tiago_moveit_vla)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(vla_cartesian_executor
  src/vla_cartesian_executor.cpp
)

ament_target_dependencies(vla_cartesian_executor
  rclcpp
  geometry_msgs
  tf2
  tf2_geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  vla_cartesian_executor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
