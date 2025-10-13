/*
 * Copyright Notes
 *
 * Authors: Yulun Tian (yulun@mit.edu), Yun Chang (yunchang@mit.edu)
 */

#include <kimera_distributed/DistributedLoopClosureRos.h>

#include <rclcpp/rclcpp.hpp>

using namespace kimera_distributed;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("kimera_distributed_loop_closure_node");

  DistributedLoopClosureRos dlcd(node);

  // multi threaded spinner
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();

  return 0;
}
