cmake_minimum_required(VERSION 3.8)
project(kimera_distributed)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pose_graph_tools_msgs REQUIRED)
find_package(pose_graph_tools_ros REQUIRED)
find_package(kimera_multi_lcd REQUIRED)
find_package(opengv REQUIRED)

find_package(Boost REQUIRED COMPONENTS
  date_time
  serialization
  thread
  filesystem
  system
  regex
  timer
  chrono
)

find_package(GTSAM REQUIRED)
find_package(DBoW2 REQUIRED)
message(STATUS "DBoW2 include dirs: ${DBoW2_INCLUDE_DIR}")
message(STATUS "DBoW2 include dirs: ${DBoW2_LIBRARY}")

# Include directories
include_directories(include)

# Create library
add_library(${PROJECT_NAME}
  src/Submap.cpp
  src/SubmapAtlas.cpp
  src/DistributedLoopClosure.cpp
  src/DistributedLoopClosureRos.cpp
  src/utils.cpp
)

# Set dependencies for the library
set(dependencies
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_filters
  image_transport
  pcl_ros
  rosbag2_cpp
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  pose_graph_tools_msgs
  pose_graph_tools_ros
  DBoW2
  kimera_multi_lcd
  opengv
)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  gtsam
  ${DBoW2_LIBRARY}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
  ${DBoW2_INCLUDE_DIR}/../
)

# Create executable
add_executable(kimera_distributed_loop_closure_node
  src/DistributedLoopClosureNode.cpp)

ament_target_dependencies(kimera_distributed_loop_closure_node ${dependencies})
target_link_libraries(kimera_distributed_loop_closure_node ${PROJECT_NAME})

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)

  # Run linting tests automatically
  ament_lint_auto_find_test_dependencies()

  # Unit tests
  ament_add_gtest(${PROJECT_NAME}-test_utils test/test_utils.cpp)

  if(TARGET ${PROJECT_NAME}-test_utils)
    target_link_libraries(${PROJECT_NAME}-test_utils ${PROJECT_NAME})
    ament_target_dependencies(${PROJECT_NAME}-test_utils ${dependencies})
  endif()

  ament_add_gtest(${PROJECT_NAME}-test_submap test/test_submap.cpp)

  if(TARGET ${PROJECT_NAME}-test_submap)
    target_link_libraries(${PROJECT_NAME}-test_submap ${PROJECT_NAME})
    ament_target_dependencies(${PROJECT_NAME}-test_submap ${dependencies})
  endif()

  ament_add_gtest(${PROJECT_NAME}-test_submap_atlas test/test_submap_atlas.cpp)

  if(TARGET ${PROJECT_NAME}-test_submap_atlas)
    target_link_libraries(${PROJECT_NAME}-test_submap_atlas ${PROJECT_NAME})
    ament_target_dependencies(${PROJECT_NAME}-test_submap_atlas ${dependencies})
  endif()
endif()

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS kimera_distributed_loop_closure_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Install launch files, config files, etc.
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.xml"
)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params
  FILES_MATCHING PATTERN "*.yaml"
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

# Export dependencies and targets
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})

ament_package()
