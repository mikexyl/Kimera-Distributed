/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu) Yulun Tian (yulun@mit.edu)
 */

#pragma once

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/path.hpp>
#include <pose_graph_tools_msgs/msg/bow_queries.hpp>
#include <pose_graph_tools_msgs/msg/bow_requests.hpp>
#include <pose_graph_tools_msgs/msg/loop_closures.hpp>
#include <pose_graph_tools_msgs/msg/loop_closures_ack.hpp>
#include <pose_graph_tools_msgs/msg/pose_graph.hpp>
#include <pose_graph_tools_msgs/msg/vlc_frames.hpp>
#include <pose_graph_tools_msgs/msg/vlc_requests.hpp>
#include <pose_graph_tools_msgs/srv/pose_graph_query.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <std_msgs/msg/u_int16_multi_array.hpp>
#include <thread>
#include <vector>

#include "kimera_distributed/DistributedLoopClosure.h"
#include "kimera_distributed/utils.h"

namespace lcd = kimera_multi_lcd;

namespace kimera_distributed {

class DistributedLoopClosureRos : DistributedLoopClosure {
 public:
  DistributedLoopClosureRos(rclcpp::Node::SharedPtr node);
  ~DistributedLoopClosureRos();

 private:
  rclcpp::Node::SharedPtr node_;
  std::atomic<bool> should_shutdown_{false};

  // ROS subscriber
  rclcpp::Subscription<pose_graph_tools_msgs::msg::PoseGraph>::SharedPtr local_pg_sub_;
  rclcpp::Subscription<pose_graph_tools_msgs::msg::VLCFrames>::SharedPtr
      internal_vlc_sub_;
  std::vector<rclcpp::Subscription<pose_graph_tools_msgs::msg::BowQueries>::SharedPtr>
      bow_sub_;
  std::vector<rclcpp::Subscription<pose_graph_tools_msgs::msg::BowRequests>::SharedPtr>
      bow_requests_sub_;
  std::vector<rclcpp::Subscription<pose_graph_tools_msgs::msg::VLCRequests>::SharedPtr>
      vlc_requests_sub_;
  std::vector<rclcpp::Subscription<pose_graph_tools_msgs::msg::VLCFrames>::SharedPtr>
      vlc_responses_sub_;
  std::vector<rclcpp::Subscription<pose_graph_tools_msgs::msg::LoopClosures>::SharedPtr>
      loop_sub_;
  std::vector<
      rclcpp::Subscription<pose_graph_tools_msgs::msg::LoopClosuresAck>::SharedPtr>
      loop_ack_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr dpgo_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt16MultiArray>::SharedPtr connectivity_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr dpgo_frame_corrector_sub_;

  // ROS publisher
  rclcpp::Publisher<pose_graph_tools_msgs::msg::VLCFrames>::SharedPtr
      vlc_responses_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::VLCRequests>::SharedPtr
      vlc_requests_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::PoseGraph>::SharedPtr pose_graph_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::BowRequests>::SharedPtr
      bow_requests_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::BowQueries>::SharedPtr
      bow_response_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::LoopClosures>::SharedPtr loop_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::LoopClosuresAck>::SharedPtr
      loop_ack_pub_;
  rclcpp::Publisher<pose_graph_tools_msgs::msg::PoseGraph>::SharedPtr
      optimized_nodes_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr optimized_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr dpgo_frame_corrector_pub_;

  // ROS service
  rclcpp::Service<pose_graph_tools_msgs::srv::PoseGraphQuery>::SharedPtr
      pose_graph_request_server_;

  // Timer
  rclcpp::TimerBase::SharedPtr log_timer_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
  rclcpp::Time start_time_;
  rclcpp::Time next_loop_sync_time_;
  rclcpp::Time next_latest_bow_pub_time_;

  // TF broadcaster from world to robot's odom frame
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

 private:
  std::string latest_kf_frame_id_;
  std::string odom_frame_id_;
  std::string world_frame_id_;

  /**
   * @brief Run place recognition / loop detection spin
   */
  void runDetection();

  /**
   * Run Geometric verification spin
   */
  void runVerification();

  /**
   * Run Comms spin to send requests
   */
  void runComms();

  /**
   * Callback to process bag of word vectors received from robots
   */
  void bowCallback(const pose_graph_tools_msgs::msg::BowQueries::SharedPtr query_msg);

  /**
   * @brief Subscribe to incremental pose graph of this robot published by VIO
   * @param msg
   */
  void localPoseGraphCallback(
      const pose_graph_tools_msgs::msg::PoseGraph::SharedPtr msg);

  /**
   * Callback to process the VLC responses to our requests
   */
  void vlcResponsesCallback(const pose_graph_tools_msgs::msg::VLCFrames::SharedPtr msg);

  /**
   * Callback to process internal VLC frames
   */
  void internalVLCCallback(const pose_graph_tools_msgs::msg::VLCFrames::SharedPtr msg);

  /**
   * @brief Callback to process the BoW requests from other robots
   * @param msg
   */
  void bowRequestsCallback(
      const pose_graph_tools_msgs::msg::BowRequests::SharedPtr msg);

  /**
   * Callback to process the VLC requests from other robots
   */
  void vlcRequestsCallback(
      const pose_graph_tools_msgs::msg::VLCRequests::SharedPtr msg);

  /**
   * Callback to process new inter-robot loop closures
   */
  void loopClosureCallback(
      const pose_graph_tools_msgs::msg::LoopClosures::SharedPtr msg);

  /**
   * Callback to process new inter-robot loop closures
   */
  void loopAcknowledgementCallback(
      const pose_graph_tools_msgs::msg::LoopClosuresAck::SharedPtr msg);

  /**
   * @brief Callback to receive optimized submap poses from dpgo
   * @param msg
   */
  void dpgoCallback(const nav_msgs::msg::Path::SharedPtr msg);

  /**
   * @brief Publish optimized nodes
   * @param msg
   */
  void publishOptimizedNodesAndPath(const gtsam::Values& nodes);

  /**
   * @brief Subscribe to T_world_dpgo (default to identity)
   * @param msg
   */
  void dpgoFrameCorrectionCallback(const geometry_msgs::msg::Pose::SharedPtr msg);

  /**
   * @brief Callback to timer used for periodically logging
   */
  void logTimerCallback();

  /**
   * @brief Publish world to dpgo frame based on first robot
   */
  void publishWorldToDpgoCorrection();

  /**
   * @brief Callback to timer used for periodically publishing TF
   */
  void tfTimerCallback();

  /**
   * @brief Subscriber callback that listens to the list of currently connected robots
   */
  void connectivityCallback(const std_msgs::msg::UInt16MultiArray::SharedPtr msg);

  /**
   * @brief Send submap-level pose graph for distributed optimization
   * @param request
   * @param response
   * @return
   */
  bool requestPoseGraphCallback(
      const pose_graph_tools_msgs::srv::PoseGraphQuery::Request::SharedPtr request,
      pose_graph_tools_msgs::srv::PoseGraphQuery::Response::SharedPtr response);

  /**
   * Initialize loop closures
   */
  void initializeLoopPublishers();

  /**
   * Publish queued loop closures to be synchronized with other robots
   */
  void publishQueuedLoops();

  /**
   * Publish VLC requests
   */
  void processVLCRequests(const size_t& robot_id,
                          const lcd::RobotPoseIdSet& vertex_ids);

  /**
   * Publish VLC Frame requests from other robots
   */
  void publishVLCRequests(const size_t& robot_id,
                          const lcd::RobotPoseIdSet& vertex_ids);

  /**
   * Request local VLC Frames from Kimera-VIO-ROS
   */
  bool requestVLCFrameService(const lcd::RobotPoseIdSet& vertex_ids);

  /**
   * @brief Request BoW vectors
   */
  void requestBowVectors();

  /**
   * @brief Publish BoW vectors requested by other robots
   */
  void publishBowVectors();

  /**
   * @brief Publish the latest BoW vector of this robot
   */
  void publishLatestBowVector();

  /**
   * Check and submit VLC requests
   */
  void requestFrames();

  /**
   * @brief Publish VLC frames requested by other robots
   */
  void publishFrames();

  void publishSubmapOfflineInfo();

  /**
   * Randomly sleep from (min_sec, max_sec) seconds
   */
  void randomSleep(double min_sec, double max_sec);

  /**
   * @brief Publish TF between world and odom
   */
  void publishOdomToWorld();

  /**
   * @brief Publish TF between world and latest keyframe
   */
  void publishLatestKFToWorld();

  /**
   * @brief Publish TF between odom and latest keyframe
   */
  void publishLatestKFToOdom();

  /**
   * @brief Save VLC frames and BoW vectors
   */
  void save();
};

}  // namespace kimera_distributed