/*
 * Copyright Notes
 *
 * Authors: Yulun Tian (yulun@mit.edu), Yun Chang (yunchang@mit.edu)
 */

#include <kimera_distributed/DistributedLoopClosureRos.h>
#include <ros/ros.h>

using namespace kimera_distributed;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::init(argc, argv, "kimera_distributed_loop_closure_node");
  ros::NodeHandle nh;

  DistributedLoopClosureRos dlcd(nh);

  ros::spin();

  return 0;
}
