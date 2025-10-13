/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu) Yulun Tian (yulun@mit.edu)
 */

#include "kimera_distributed/DistributedLoopClosureRos.h"

#include <DBoW2/DBoW2.h>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <kimera_multi_lcd/utils.h>
#include <pose_graph_tools_ros/utils.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <pose_graph_tools_msgs/msg/pose_graph.hpp>
#include <pose_graph_tools_msgs/srv/vlc_frame_query.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "kimera_distributed/configs.h"

namespace kimera_distributed {

DistributedLoopClosureRos::DistributedLoopClosureRos(rclcpp::Node::SharedPtr node)
    : node_(node) {
  DistributedLoopClosureConfig config;
  int my_id_int = -1;
  int num_robots_int = -1;
  my_id_int = node_->declare_parameter("robot_id", -1);
  num_robots_int = node_->declare_parameter("num_robots", -1);
  config.frame_id_ = node_->declare_parameter("frame_id", std::string("world"));
  assert(my_id_int >= 0);
  assert(num_robots_int > 0);
  config.my_id_ = my_id_int;
  config.num_robots_ = num_robots_int;

  // Path to log outputs
  config.log_output_dir_ = node_->declare_parameter("log_output_path", std::string(""));
  config.log_output_ = !config.log_output_dir_.empty();
  config.run_offline_ = node_->declare_parameter("run_offline", false);

  if (config.run_offline_) {
    config.offline_dir_ = node_->declare_parameter("offline_dir", std::string(""));
    if (config.offline_dir_.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "Offline directory is missing!");
      rclcpp::shutdown();
    }
  }

  // Visual place recognition params
  config.lcd_params_.alpha_ = node_->declare_parameter("alpha", 0.1);
  config.lcd_params_.dist_local_ = node_->declare_parameter("dist_local", 3);
  config.lcd_params_.max_db_results_ = node_->declare_parameter("max_db_results", 50);
  config.lcd_params_.min_nss_factor_ =
      node_->declare_parameter("min_nss_factor", 0.005);

  // Lcd Third Party Wrapper Params
  config.lcd_params_.lcd_tp_params_.max_nrFrames_between_islands_ =
      node_->declare_parameter("max_nrFrames_between_islands", 3);
  config.lcd_params_.lcd_tp_params_.max_nrFrames_between_queries_ =
      node_->declare_parameter("max_nrFrames_between_queries", 10);
  config.lcd_params_.lcd_tp_params_.max_intraisland_gap_ =
      node_->declare_parameter("max_intraisland_gap", 5);
  config.lcd_params_.lcd_tp_params_.min_matches_per_island_ =
      node_->declare_parameter("min_matches_per_island", 3);
  config.lcd_params_.lcd_tp_params_.min_temporal_matches_ =
      node_->declare_parameter("min_temporal_matches", 3);

  // Geometric verification params
  config.lcd_params_.ransac_threshold_mono_ =
      node_->declare_parameter("ransac_threshold_mono", 1e-6);
  config.lcd_params_.ransac_inlier_percentage_mono_ =
      node_->declare_parameter("ransac_inlier_percentage_mono", 0.1);
  config.lcd_params_.max_ransac_iterations_mono_ =
      node_->declare_parameter("max_ransac_iterations_mono", 100);
  config.lcd_params_.lowe_ratio_ = node_->declare_parameter("lowe_ratio", 0.8);
  config.lcd_params_.max_ransac_iterations_ =
      node_->declare_parameter("max_ransac_iterations", 1000);
  config.lcd_params_.ransac_threshold_ =
      node_->declare_parameter("ransac_threshold", 1e-4);
  config.lcd_params_.geometric_verification_min_inlier_count_ =
      node_->declare_parameter("geometric_verification_min_inlier_count", 10);
  config.lcd_params_.geometric_verification_min_inlier_percentage_ =
      node_->declare_parameter("geometric_verification_min_inlier_percentage", 0.12);
  config.lcd_params_.inter_robot_only_ =
      node_->declare_parameter("detect_interrobot_only", false);

  config.lcd_params_.vocab_path_ =
      node_->declare_parameter("vocabulary_path", std::string(""));

  config.detection_batch_size_ = node_->declare_parameter("detection_batch_size", 50);
  config.bow_skip_num_ = node_->declare_parameter("bow_skip_num", 0);
  // Load parameters controlling VLC communication
  config.bow_batch_size_ = node_->declare_parameter("bow_batch_size", 10);
  config.vlc_batch_size_ = node_->declare_parameter("vlc_batch_size", 10);
  config.loop_batch_size_ = node_->declare_parameter("loop_batch_size", 5);
  config.comm_sleep_time_ = node_->declare_parameter("comm_sleep_time", 1.0);
  config.loop_sync_sleep_time_ = node_->declare_parameter("loop_sync_sleep_time", 5.0);

  // TF
  latest_kf_frame_id_ = node_->declare_parameter("latest_kf_frame_id", std::string(""));
  if (latest_kf_frame_id_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Latest KF frame ID is missing!");
    rclcpp::shutdown();
  }
  odom_frame_id_ = node_->declare_parameter("odom_frame_id", std::string(""));
  if (odom_frame_id_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Odometry frame ID is missing!");
    rclcpp::shutdown();
  }
  world_frame_id_ = node_->declare_parameter("world_frame_id", std::string(""));
  if (world_frame_id_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "World frame ID is missing!");
    rclcpp::shutdown();
  }

  // Load robot names and initialize candidate lc queues
  for (size_t id = 0; id < config.num_robots_; id++) {
    std::string robot_name = "kimera" + std::to_string(id);
    robot_name =
        node_->declare_parameter("robot" + std::to_string(id) + "_name", robot_name);
    config.robot_names_[id] = robot_name;
  }

  config.submap_params_.max_submap_size =
      node_->declare_parameter("max_submap_size", 10);
  config.submap_params_.max_submap_distance =
      node_->declare_parameter("max_submap_distance", 5.0);

  initialize(config);

  // Subscriber
  std::string topic = "/" + config_.robot_names_[config_.my_id_] +
                      "/kimera_vio_ros/pose_graph_incremental";
  local_pg_sub_ = node_->create_subscription<pose_graph_tools_msgs::msg::PoseGraph>(
      topic,
      1000,
      std::bind(&DistributedLoopClosureRos::localPoseGraphCallback,
                this,
                std::placeholders::_1));

  std::string internal_vlc_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/vlc_frames";
  internal_vlc_sub_ = node_->create_subscription<pose_graph_tools_msgs::msg::VLCFrames>(
      internal_vlc_topic,
      1000,
      std::bind(&DistributedLoopClosureRos::internalVLCCallback,
                this,
                std::placeholders::_1));

  std::string dpgo_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/dpgo_ros_node/path";
  dpgo_sub_ = node_->create_subscription<nav_msgs::msg::Path>(
      dpgo_topic,
      3,
      std::bind(&DistributedLoopClosureRos::dpgoCallback, this, std::placeholders::_1));

  std::string connectivity_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/connected_peer_ids";
  connectivity_sub_ = node_->create_subscription<std_msgs::msg::UInt16MultiArray>(
      connectivity_topic,
      5,
      std::bind(&DistributedLoopClosureRos::connectivityCallback,
                this,
                std::placeholders::_1));

  for (size_t id = 0; id < config_.num_robots_; ++id) {
    if (id < config_.my_id_) {
      std::string vlc_req_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/vlc_requests";
      auto vlc_req_sub =
          node_->create_subscription<pose_graph_tools_msgs::msg::VLCRequests>(
              vlc_req_topic,
              1,
              std::bind(&DistributedLoopClosureRos::vlcRequestsCallback,
                        this,
                        std::placeholders::_1));
      vlc_requests_sub_.push_back(vlc_req_sub);

      std::string bow_req_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/bow_requests";
      auto bow_req_sub =
          node_->create_subscription<pose_graph_tools_msgs::msg::BowRequests>(
              bow_req_topic,
              1,
              std::bind(&DistributedLoopClosureRos::bowRequestsCallback,
                        this,
                        std::placeholders::_1));
      bow_requests_sub_.push_back(bow_req_sub);

      std::string loop_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/loop_closures";
      auto loop_sub =
          node_->create_subscription<pose_graph_tools_msgs::msg::LoopClosures>(
              loop_topic,
              100,
              std::bind(&DistributedLoopClosureRos::loopClosureCallback,
                        this,
                        std::placeholders::_1));
      loop_sub_.push_back(loop_sub);
    }

    if (id >= config_.my_id_) {
      std::string bow_topic =
          "/" + config_.robot_names_[id] + "/kimera_vio_ros/bow_query";
      auto bow_sub = node_->create_subscription<pose_graph_tools_msgs::msg::BowQueries>(
          bow_topic,
          1000,
          std::bind(
              &DistributedLoopClosureRos::bowCallback, this, std::placeholders::_1));
      bow_sub_.push_back(bow_sub);
      bow_latest_[id] = 0;
      bow_received_[id] = std::unordered_set<lcd::PoseId>();
    }

    if (id > config_.my_id_) {
      std::string resp_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/vlc_responses";
      auto resp_sub = node_->create_subscription<pose_graph_tools_msgs::msg::VLCFrames>(
          resp_topic,
          10,
          std::bind(&DistributedLoopClosureRos::vlcResponsesCallback,
                    this,
                    std::placeholders::_1));
      vlc_responses_sub_.push_back(resp_sub);

      std::string ack_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/loop_ack";
      auto ack_sub =
          node_->create_subscription<pose_graph_tools_msgs::msg::LoopClosuresAck>(
              ack_topic,
              100,
              std::bind(&DistributedLoopClosureRos::loopAcknowledgementCallback,
                        this,
                        std::placeholders::_1));
      loop_ack_sub_.push_back(ack_sub);
    }
  }

  if (config_.my_id_ > 0) {
    std::string pose_corrector_topic =
        "/" + config_.robot_names_[0] + "/kimera_distributed/pose_corrector";
    dpgo_frame_corrector_sub_ = node_->create_subscription<geometry_msgs::msg::Pose>(
        pose_corrector_topic,
        1,
        std::bind(&DistributedLoopClosureRos::dpgoFrameCorrectionCallback,
                  this,
                  std::placeholders::_1));
  }

  // Publisher
  std::string bow_response_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/bow_query";
  bow_response_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::BowQueries>(
      bow_response_topic, 1000);

  std::string bow_request_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/bow_requests";
  bow_requests_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::BowRequests>(
      bow_request_topic, 100);

  std::string pose_graph_topic = "/" + config_.robot_names_[config_.my_id_] +
                                 "/kimera_distributed/pose_graph_incremental";
  pose_graph_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::PoseGraph>(
      pose_graph_topic, 1000);

  std::string resp_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/vlc_responses";
  vlc_responses_pub_ =
      node_->create_publisher<pose_graph_tools_msgs::msg::VLCFrames>(resp_topic, 10);

  std::string req_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/vlc_requests";
  vlc_requests_pub_ =
      node_->create_publisher<pose_graph_tools_msgs::msg::VLCRequests>(req_topic, 10);

  std::string loop_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/loop_closures";
  loop_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::LoopClosures>(
      loop_topic, 100);

  std::string ack_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/loop_ack";
  loop_ack_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::LoopClosuresAck>(
      ack_topic, 100);

  std::string optimized_nodes_topic = "/" + config_.robot_names_[config_.my_id_] +
                                      "/kimera_distributed/optimized_nodes";
  optimized_nodes_pub_ = node_->create_publisher<pose_graph_tools_msgs::msg::PoseGraph>(
      optimized_nodes_topic, 1);

  std::string optimized_path_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/optimized_path";
  optimized_path_pub_ =
      node_->create_publisher<nav_msgs::msg::Path>(optimized_path_topic, 1);

  if (config_.my_id_ == 0) {
    std::string pose_corrector_topic = "/" + config_.robot_names_[config_.my_id_] +
                                       "/kimera_distributed/pose_corrector";
    dpgo_frame_corrector_pub_ =
        node_->create_publisher<geometry_msgs::msg::Pose>(pose_corrector_topic, 1);
  }

  // Initialize TF broadcaster
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

  // ROS service
  pose_graph_request_server_ =
      node_->create_service<pose_graph_tools_msgs::srv::PoseGraphQuery>(
          "request_pose_graph",
          std::bind(&DistributedLoopClosureRos::requestPoseGraphCallback,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2));

  log_timer_ = node_->create_wall_timer(
      std::chrono::seconds(10),
      std::bind(&DistributedLoopClosureRos::logTimerCallback, this));

  tf_timer_ = node_->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&DistributedLoopClosureRos::tfTimerCallback, this));

  RCLCPP_INFO_STREAM(
      node_->get_logger(),
      "Distributed Kimera node initialized (ID = "
          << config_.my_id_ << "). \n"
          << "Parameters: \n"
          << "alpha = " << config_.lcd_params_.alpha_ << "\n"
          << "dist_local = " << config_.lcd_params_.dist_local_ << "\n"
          << "max_db_results = " << config_.lcd_params_.max_db_results_ << "\n"
          << "min_nss_factor = " << config_.lcd_params_.min_nss_factor_ << "\n"
          << "lowe_ratio = " << config_.lcd_params_.lowe_ratio_ << "\n"
          << "max_nrFrames_between_queries = "
          << config_.lcd_params_.lcd_tp_params_.max_nrFrames_between_queries_ << "\n"
          << "max_nrFrames_between_islands = "
          << config_.lcd_params_.lcd_tp_params_.max_nrFrames_between_islands_ << "\n"
          << "max_intraisland_gap = "
          << config_.lcd_params_.lcd_tp_params_.max_intraisland_gap_ << "\n"
          << "min_matches_per_island = "
          << config_.lcd_params_.lcd_tp_params_.min_matches_per_island_ << "\n"
          << "min_temporal_matches = "
          << config_.lcd_params_.lcd_tp_params_.min_temporal_matches_ << "\n"
          << "max_ransac_iterations = " << config_.lcd_params_.max_ransac_iterations_
          << "\n"
          << "mono ransac threshold = " << config_.lcd_params_.ransac_threshold_mono_
          << "\n"
          << "mono ransac max iterations = "
          << config_.lcd_params_.max_ransac_iterations_mono_ << "\n"
          << "mono ransac min inlier percentage = "
          << config_.lcd_params_.ransac_inlier_percentage_mono_ << "\n"
          << "ransac_threshold = " << config_.lcd_params_.ransac_threshold_ << "\n"
          << "geometric_verification_min_inlier_count = "
          << config_.lcd_params_.geometric_verification_min_inlier_count_ << "\n"
          << "geometric_verification_min_inlier_percentage = "
          << config_.lcd_params_.geometric_verification_min_inlier_percentage_ << "\n"
          << "interrobot loop closure only = " << config_.lcd_params_.inter_robot_only_
          << "\n"
          << "maximum batch size to request BoW vectors = " << config_.bow_batch_size_
          << "\n"
          << "maximum batch size to request VLC frames = " << config_.vlc_batch_size_
          << "\n"
          << "Communication thread sleep time = " << config_.comm_sleep_time_ << "\n"
          << "maximum submap size = " << config_.submap_params_.max_submap_size << "\n"
          << "maximum submap distance = " << config_.submap_params_.max_submap_distance
          << "\n"
          << "loop detection batch size = " << config_.detection_batch_size_ << "\n"
          << "loop synchronization batch size = " << config_.loop_batch_size_ << "\n"
          << "loop synchronization sleep time = " << config_.loop_sync_sleep_time_
          << "\n"
          << "BoW vector skip num = " << config_.bow_skip_num_ << "\n");

  if (config_.run_offline_) {
    // publish submap poses
    for (int count = 0; count < 3; ++count) {
      publishSubmapOfflineInfo();
    }
    processOfflineLoopClosures();
  }

  // Start loop detection thread
  detection_thread_.reset(
      new std::thread(&DistributedLoopClosureRos::runDetection, this));
  RCLCPP_INFO(node_->get_logger(),
              "Robot %zu started loop detection / place recognition thread.",
              config_.my_id_);

  // Start verification thread
  verification_thread_.reset(
      new std::thread(&DistributedLoopClosureRos::runVerification, this));
  RCLCPP_INFO(node_->get_logger(),
              "Robot %zu started loop verification thread.",
              config_.my_id_);

  // Start comms thread
  comms_thread_.reset(new std::thread(&DistributedLoopClosureRos::runComms, this));
  RCLCPP_INFO(
      node_->get_logger(), "Robot %zu started communication thread.", config_.my_id_);

  start_time_ = node_->get_clock()->now();
  next_loop_sync_time_ = node_->get_clock()->now();
  next_latest_bow_pub_time_ = node_->get_clock()->now();
}

DistributedLoopClosureRos::~DistributedLoopClosureRos() {
  if (config_.log_output_) {
    save();
  }
  RCLCPP_INFO(node_->get_logger(),
              "Shutting down DistributedLoopClosureRos process on robot %zu...",
              config_.my_id_);
}

void DistributedLoopClosureRos::bowCallback(
    const pose_graph_tools_msgs::msg::BowQueries::SharedPtr query_msg) {
  processBow(query_msg);
}

void DistributedLoopClosureRos::localPoseGraphCallback(
    const pose_graph_tools_msgs::msg::PoseGraph::SharedPtr msg) {
  bool incremental_pub = processLocalPoseGraph(msg);

  // Publish sparsified pose graph
  pose_graph_tools_msgs::msg::PoseGraph sparse_pose_graph =
      getSubmapPoseGraph(incremental_pub);
  if (!sparse_pose_graph.edges.empty() || !sparse_pose_graph.nodes.empty()) {
    pose_graph_pub_->publish(sparse_pose_graph);
  }
}

void DistributedLoopClosureRos::connectivityCallback(
    const std_msgs::msg::UInt16MultiArray::SharedPtr msg) {
  std::set<unsigned> connected_ids(msg->data.begin(), msg->data.end());
  for (unsigned robot_id = 0; robot_id < config_.num_robots_; ++robot_id) {
    if (robot_id == config_.my_id_) {
      robot_connected_[robot_id] = true;
    } else if (connected_ids.find(robot_id) != connected_ids.end()) {
      robot_connected_[robot_id] = true;
    } else {
      // ROS_WARN("Robot %u is disconnected.", robot_id);
      robot_connected_[robot_id] = false;
    }
  }
}

void DistributedLoopClosureRos::dpgoCallback(const nav_msgs::msg::Path::SharedPtr msg) {
  processOptimizedPath(msg);
  gtsam::Values::shared_ptr nodes_ptr(new gtsam::Values);
  computePosesInWorldFrame(nodes_ptr);
  if (config_.run_offline_) {
    saveSubmapAtlas(config_.log_output_dir_);
    auto elapsed_time = node_->now() - start_time_;
    int elapsed_sec = int(elapsed_time.seconds());
    std::string file_path = config_.log_output_dir_ + "kimera_distributed_poses_" +
                            std::to_string(elapsed_sec) + ".csv";
    savePosesToFile(file_path, *nodes_ptr);
  }
  publishOptimizedNodesAndPath(*nodes_ptr);
}

void DistributedLoopClosureRos::publishOptimizedNodesAndPath(
    const gtsam::Values& nodes) {
  pose_graph_tools_msgs::msg::PoseGraph nodes_msg;
  nav_msgs::msg::Path path_msg;
  for (const auto& key_pose : nodes) {
    pose_graph_tools_msgs::msg::PoseGraphNode node_msg;
    gtsam::Symbol key_symb(key_pose.key);
    node_msg.key = key_symb.index();
    node_msg.robot_id = config_.my_id_;
    node_msg.pose = GtsamPoseToRos(nodes.at<gtsam::Pose3>(node_msg.key));
    const auto keyframe = submap_atlas_->getKeyframe(key_pose.key);
    node_msg.header.stamp = rclcpp::Time(keyframe->stamp());
    nodes_msg.nodes.push_back(node_msg);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.pose = node_msg.pose;
    pose_stamped.header = node_msg.header;
    path_msg.poses.push_back(pose_stamped);
    path_msg.header = node_msg.header;
  }
  path_msg.header.frame_id = world_frame_id_;
  optimized_nodes_pub_->publish(nodes_msg);
  optimized_path_pub_->publish(path_msg);
}

void DistributedLoopClosureRos::dpgoFrameCorrectionCallback(
    const geometry_msgs::msg::Pose::SharedPtr msg) {
  T_world_dpgo_ = RosPoseToGtsam(*msg);
}

void DistributedLoopClosureRos::logTimerCallback() {
  if (!config_.log_output_) return;
  if (config_.run_offline_) return;
  logLcdStat();
  // Save latest submap atlas
  saveSubmapAtlas(config_.log_output_dir_);
  // Save latest trajectory estimates in the world frame
  if (backend_update_count_ > 0) {
    auto elapsed_time = node_->now() - start_time_;
    int elapsed_sec = int(elapsed_time.seconds());
    std::string file_path = config_.log_output_dir_ + "kimera_distributed_poses_" +
                            std::to_string(elapsed_sec) + ".csv";
    gtsam::Values::shared_ptr nodes_ptr(new gtsam::Values);
    computePosesInWorldFrame(nodes_ptr);
    savePosesToFile(file_path, *nodes_ptr);
  }
}

void DistributedLoopClosureRos::publishWorldToDpgoCorrection() {
  if (config_.my_id_ != 0) {
    return;
  }

  dpgo_frame_corrector_pub_->publish(GtsamPoseToRos(T_world_dpgo_));
}

void DistributedLoopClosureRos::publishOdomToWorld() {
  geometry_msgs::msg::TransformStamped tf_world_odom;
  tf_world_odom.header.stamp = node_->now();
  tf_world_odom.header.frame_id = world_frame_id_;
  tf_world_odom.child_frame_id = odom_frame_id_;

  const gtsam::Pose3 T_world_odom = getOdomInWorldFrame();
  GtsamPoseToRosTf(T_world_odom, &tf_world_odom.transform);
  tf_broadcaster_->sendTransform(tf_world_odom);
}

void DistributedLoopClosureRos::publishLatestKFToWorld() {
  geometry_msgs::msg::TransformStamped tf_world_base;
  tf_world_base.header.stamp = node_->now();
  tf_world_base.header.frame_id = world_frame_id_;
  tf_world_base.child_frame_id = latest_kf_frame_id_;

  const gtsam::Pose3 T_world_base = getLatestKFInWorldFrame();
  GtsamPoseToRosTf(T_world_base, &tf_world_base.transform);
  tf_broadcaster_->sendTransform(tf_world_base);
}

void DistributedLoopClosureRos::publishLatestKFToOdom() {
  geometry_msgs::msg::TransformStamped tf_odom_base;
  tf_odom_base.header.stamp = node_->now();
  tf_odom_base.header.frame_id = odom_frame_id_;
  tf_odom_base.child_frame_id = latest_kf_frame_id_;

  const gtsam::Pose3 T_odom_base = getLatestKFInOdomFrame();
  GtsamPoseToRosTf(T_odom_base, &tf_odom_base.transform);
  tf_broadcaster_->sendTransform(tf_odom_base);
}

void DistributedLoopClosureRos::tfTimerCallback() {
  publishWorldToDpgoCorrection();
  publishOdomToWorld();
  publishLatestKFToOdom();  // Currently for debugging
}

void DistributedLoopClosureRos::runDetection() {
  while (rclcpp::ok() && !should_shutdown_) {
    if (!bow_msgs_.empty()) {
      detectLoopSpin();
    }
    rclcpp::sleep_for(std::chrono::seconds(1));
  }
}

void DistributedLoopClosureRos::runVerification() {
  rclcpp::WallRate r(1);
  while (rclcpp::ok() && !should_shutdown_) {
    if (queued_lc_.empty()) {
      r.sleep();
    } else {
      verifyLoopSpin();
    }
  }
}

void DistributedLoopClosureRos::runComms() {
  while (rclcpp::ok() && !should_shutdown_) {
    // Request missing Bow vectors from other robots
    requestBowVectors();

    // Request VLC frames from other robots
    size_t total_candidates = updateCandidateList();
    if (total_candidates > 0) {
      requestFrames();
    }

    // Publish BoW vectors requested by other robots
    publishBowVectors();

    // Publish VLC frames requested by other robots
    publishFrames();

    // Publish new inter-robot loop closures, if any
    if (node_->now().seconds() > next_loop_sync_time_.seconds()) {
      initializeLoopPublishers();
      publishQueuedLoops();

      // Generate a random sleep time
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> distribution(
          0.5 * config_.loop_sync_sleep_time_, 1.5 * config_.loop_sync_sleep_time_);
      double sleep_time = distribution(gen);
      next_loop_sync_time_ +=
          rclcpp::Duration::from_seconds(sleep_time);  // TODO: make into parameter
    }

    // Once a while publish latest BoW vector
    // This is needed for other robots to request potentially missing BoW
    if (node_->now().seconds() > next_latest_bow_pub_time_.seconds()) {
      publishLatestBowVector();
      next_latest_bow_pub_time_ += rclcpp::Duration::from_seconds(30);
    }

    // Print stats
    RCLCPP_INFO_STREAM(node_->get_logger(),
                       "Total inter-robot loop closures: " << num_inter_robot_loops_);

    double avg_sleep_time = (double)config_.comm_sleep_time_;
    double min_sleep_time = 0.5 * avg_sleep_time;
    double max_sleep_time = 1.5 * avg_sleep_time;
    randomSleep(min_sleep_time, max_sleep_time);
  }
}

void DistributedLoopClosureRos::requestBowVectors() {
  // Form BoW vectors that are missing from each robot
  lcd::RobotId robot_id_to_query;
  std::set<lcd::PoseId> missing_bow_vectors;
  if (!queryBowVectorsRequest(robot_id_to_query, missing_bow_vectors)) {
    return;
  }

  // Publish BoW request to selected robot
  pose_graph_tools_msgs::msg::BowRequests msg;
  msg.source_robot_id = config_.my_id_;
  msg.destination_robot_id = robot_id_to_query;
  for (const auto& pose_id : missing_bow_vectors) {
    if (msg.pose_ids.size() >= static_cast<size_t>(config_.bow_batch_size_)) break;
    msg.pose_ids.push_back(pose_id);
  }
  RCLCPP_WARN(node_->get_logger(),
              "Processing %lu BoW requests to robot %lu.",
              msg.pose_ids.size(),
              robot_id_to_query);
  bow_requests_pub_->publish(msg);
}

void DistributedLoopClosureRos::publishBowVectors() {
  lcd::RobotId robot_id_to_publish;
  lcd::RobotPoseIdSet bow_vectors_to_publish;
  if (!queryBowVectorsPublish(robot_id_to_publish, bow_vectors_to_publish)) {
    return;
  }
  // Send BoW vectors to selected robot
  pose_graph_tools_msgs::msg::BowQueries msg;
  msg.destination_robot_id = robot_id_to_publish;
  for (const auto& robot_pose_id : bow_vectors_to_publish) {
    pose_graph_tools_msgs::msg::BowQuery query_msg;
    query_msg.robot_id = robot_pose_id.first;
    query_msg.pose_id = robot_pose_id.second;
    kimera_multi_lcd::BowVectorToMsg(lcd_->getBoWVector(robot_pose_id),
                                     &(query_msg.bow_vector));
    msg.queries.push_back(query_msg);
  }
  bow_response_pub_->publish(msg);
  RCLCPP_INFO(node_->get_logger(),
              "Published %zu BoWs to robot %zu (%zu waiting).",
              msg.queries.size(),
              robot_id_to_publish,
              requested_bows_[robot_id_to_publish].size());
}

void DistributedLoopClosureRos::publishLatestBowVector() {
  int pose_id = lcd_->latestPoseIdWithBoW(config_.my_id_);
  if (pose_id != -1) {
    lcd::RobotPoseId latest_id(config_.my_id_, pose_id);
    pose_graph_tools_msgs::msg::BowQuery query_msg;
    query_msg.robot_id = config_.my_id_;
    query_msg.pose_id = pose_id;
    kimera_multi_lcd::BowVectorToMsg(lcd_->getBoWVector(latest_id),
                                     &(query_msg.bow_vector));

    pose_graph_tools_msgs::msg::BowQueries msg;
    msg.queries.push_back(query_msg);
    for (lcd::RobotId robot_id = 0; robot_id < config_.my_id_; ++robot_id) {
      msg.destination_robot_id = robot_id;
      bow_response_pub_->publish(msg);
    }
  }
  RCLCPP_INFO(node_->get_logger(), "Published latest BoW vector.");
}

void DistributedLoopClosureRos::requestFrames() {
  lcd::RobotPoseIdSet my_vertex_ids, target_vertex_ids;
  lcd::RobotId target_robot_id;
  queryFramesRequest(my_vertex_ids, target_robot_id, target_vertex_ids);

  // Process missing VLC frames of myself
  if (my_vertex_ids.size() > 0) {
    processVLCRequests(config_.my_id_, my_vertex_ids);
  }

  if (target_vertex_ids.size() > 0) {
    processVLCRequests(target_robot_id, target_vertex_ids);
  }
}

void DistributedLoopClosureRos::publishFrames() {
  lcd::RobotId target_robot_id;
  lcd::RobotPoseIdSet target_vertex_ids;
  queryFramesPublish(target_robot_id, target_vertex_ids);

  if (target_vertex_ids.size() > 0) {
    // Send VLC frames to the selected robot
    pose_graph_tools_msgs::msg::VLCFrames frames_msg;
    frames_msg.destination_robot_id = target_robot_id;
    for (const auto& vertex_id : target_vertex_ids) {
      pose_graph_tools_msgs::msg::VLCFrameMsg vlc_msg;
      kimera_multi_lcd::VLCFrameToMsg(lcd_->getVLCFrame(vertex_id), &vlc_msg);
      frames_msg.frames.push_back(vlc_msg);
    }
    vlc_responses_pub_->publish(frames_msg);
    RCLCPP_INFO(node_->get_logger(),
                "Published %zu frames to robot %zu (%zu frames waiting).",
                frames_msg.frames.size(),
                target_robot_id,
                requested_frames_[target_robot_id].size());
  }
}

bool DistributedLoopClosureRos::requestPoseGraphCallback(
    const pose_graph_tools_msgs::srv::PoseGraphQuery::Request::SharedPtr request,
    pose_graph_tools_msgs::srv::PoseGraphQuery::Response::SharedPtr response) {
  CHECK_EQ(request->robot_id, config_.my_id_);
  response->pose_graph = getSubmapPoseGraph();

  if (config_.run_offline_) {
    if (!offline_keyframe_loop_closures_.empty()) {
      RCLCPP_WARN(
          node_->get_logger(),
          "[requestPoseGraphCallback] %zu offline loop closures not yet processed.",
          offline_keyframe_loop_closures_.size());
      return false;
    }
    if (!submap_loop_closures_queue_.empty()) {
      RCLCPP_WARN(node_->get_logger(),
                  "[requestPoseGraphCallback] %zu loop closures to be synchronized.",
                  submap_loop_closures_queue_.size());
      return false;
    }
  }

  return true;
}

void DistributedLoopClosureRos::initializeLoopPublishers() {
  // Publish empty loops and acks
  pose_graph_tools_msgs::msg::LoopClosures loop_msg;
  pose_graph_tools_msgs::msg::LoopClosuresAck ack_msg;
  loop_msg.publishing_robot_id = config_.my_id_;
  ack_msg.publishing_robot_id = config_.my_id_;
  for (lcd::RobotId robot_id = 0; robot_id < config_.num_robots_; ++robot_id) {
    if (robot_id != config_.my_id_ && !loop_pub_initialized_[robot_id] &&
        robot_connected_[robot_id]) {
      loop_msg.destination_robot_id = robot_id;
      loop_pub_->publish(loop_msg);

      ack_msg.destination_robot_id = robot_id;
      loop_ack_pub_->publish(ack_msg);

      loop_pub_initialized_[robot_id] = true;
    }
  }
}

void DistributedLoopClosureRos::publishQueuedLoops() {
  std::map<lcd::RobotId, size_t> robot_queue_sizes;
  std::map<lcd::RobotId, pose_graph_tools_msgs::msg::LoopClosures> msg_map;
  auto it = submap_loop_closures_queue_.begin();
  lcd::RobotId other_robot = 0;
  while (it != submap_loop_closures_queue_.end()) {
    const lcd::EdgeID& edge_id = it->first;
    const auto& factor = it->second;
    if (edge_id.robot_src == config_.my_id_) {
      other_robot = edge_id.robot_dst;
    } else {
      other_robot = edge_id.robot_src;
    }
    if (other_robot == config_.my_id_) {
      // This is a intra-robot loop closure and no need to synchronize
      submap_loop_closures_.add(factor);
      it = submap_loop_closures_queue_.erase(it);
    } else {
      // This is a inter-robot loop closure
      if (robot_queue_sizes.find(other_robot) == robot_queue_sizes.end()) {
        robot_queue_sizes[other_robot] = 0;
        pose_graph_tools_msgs::msg::LoopClosures msg;
        msg.publishing_robot_id = config_.my_id_;
        msg.destination_robot_id = other_robot;
        msg_map[other_robot] = msg;
      }
      robot_queue_sizes[other_robot]++;
      if (msg_map[other_robot].edges.size() < config_.loop_batch_size_) {
        pose_graph_tools_msgs::msg::PoseGraphEdge edge_msg;
        edge_msg.robot_from = edge_id.robot_src;
        edge_msg.robot_to = edge_id.robot_dst;
        edge_msg.key_from = edge_id.frame_src;
        edge_msg.key_to = edge_id.frame_dst;
        edge_msg.pose = GtsamPoseToRos(factor.measured());
        // TODO: write covariance
        msg_map[other_robot].edges.push_back(edge_msg);
      }
      ++it;
    }
  }

  // Select the connected robot with largest queue size to synchronize
  lcd::RobotId selected_robot_id = 0;
  size_t selected_queue_size = 0;
  for (auto& it : robot_queue_sizes) {
    lcd::RobotId robot_id = it.first;
    size_t queue_size = it.second;
    if (robot_connected_[robot_id] && queue_size >= selected_queue_size) {
      selected_robot_id = robot_id;
      selected_queue_size = queue_size;
    }
  }
  if (selected_queue_size > 0) {
    RCLCPP_WARN(node_->get_logger(),
                "Published %zu loops to robot %zu.",
                msg_map[selected_robot_id].edges.size(),
                selected_robot_id);
    loop_pub_->publish(msg_map[selected_robot_id]);
  }
}

void DistributedLoopClosureRos::loopClosureCallback(
    const pose_graph_tools_msgs::msg::LoopClosures::SharedPtr msg) {
  if (msg->destination_robot_id != config_.my_id_) {
    return;
  }
  size_t loops_added = 0;
  pose_graph_tools_msgs::msg::LoopClosuresAck ack_msg;
  ack_msg.publishing_robot_id = config_.my_id_;
  ack_msg.destination_robot_id = msg->publishing_robot_id;
  for (const auto& edge : msg->edges) {
    const lcd::EdgeID submap_edge_id(
        edge.robot_from, edge.key_from, edge.robot_to, edge.key_to);
    // For each incoming loop closure, only add locally if does not exist
    bool edge_exists = (submap_loop_closures_ids_.find(submap_edge_id) !=
                        submap_loop_closures_ids_.end());
    if (!edge_exists) {
      loops_added++;
      submap_loop_closures_ids_.emplace(submap_edge_id);
      gtsam::Symbol submap_from(robot_id_to_prefix.at(edge.robot_from), edge.key_from);
      gtsam::Symbol submap_to(robot_id_to_prefix.at(edge.robot_to), edge.key_to);
      const auto pose = RosPoseToGtsam(edge.pose);
      // TODO: read covariance from message
      static const gtsam::SharedNoiseModel& noise =
          gtsam::noiseModel::Isotropic::Variance(6, 1e-2);
      submap_loop_closures_.add(
          gtsam::BetweenFactor<gtsam::Pose3>(submap_from, submap_to, pose, noise));
    }
    // Always acknowledge all received loops
    ack_msg.robot_src.push_back(edge.robot_from);
    ack_msg.robot_dst.push_back(edge.robot_to);
    ack_msg.frame_src.push_back(edge.key_from);
    ack_msg.frame_dst.push_back(edge.key_to);
  }
  RCLCPP_WARN(node_->get_logger(),
              "Received %zu loop closures from robot %i (%zu new).",
              msg->edges.size(),
              msg->publishing_robot_id,
              loops_added);
  loop_ack_pub_->publish(ack_msg);
}

void DistributedLoopClosureRos::loopAcknowledgementCallback(
    const pose_graph_tools_msgs::msg::LoopClosuresAck::SharedPtr msg) {
  if (msg->destination_robot_id != config_.my_id_) {
    return;
  }
  size_t loops_acked = 0;
  for (size_t i = 0; i < msg->robot_src.size(); ++i) {
    const lcd::EdgeID edge_id(
        msg->robot_src[i], msg->frame_src[i], msg->robot_dst[i], msg->frame_dst[i]);
    if (submap_loop_closures_queue_.find(edge_id) !=
        submap_loop_closures_queue_.end()) {
      loops_acked++;
      // Move acknowledged loop closure from queue to factor graph
      // which will be shared with the back-end
      submap_loop_closures_.add(submap_loop_closures_queue_.at(edge_id));
      submap_loop_closures_queue_.erase(edge_id);
    }
  }
  if (loops_acked > 0) {
    RCLCPP_WARN(node_->get_logger(),
                "Received %zu loop acks from robot %i.",
                loops_acked,
                msg->publishing_robot_id);
  }
}

void DistributedLoopClosureRos::processVLCRequests(
    const size_t& robot_id,
    const lcd::RobotPoseIdSet& vertex_ids) {
  if (vertex_ids.size() == 0) {
    return;
  }

  // RCLCPP_INFO(node_->get_logger(),"Processing %lu VLC requests to robot %lu.",
  // vertex_ids.size(), robot_id);
  if (robot_id == config_.my_id_) {
    // Directly request from Kimera-VIO-ROS
    {  // start vlc service critical section
      std::unique_lock<std::mutex> service_lock(vlc_service_mutex_);
      if (!requestVLCFrameService(vertex_ids)) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Failed to retrieve local VLC frames on robot %zu.",
                     config_.my_id_);
      }
    }
  } else {
    publishVLCRequests(robot_id, vertex_ids);
  }
}

void DistributedLoopClosureRos::publishVLCRequests(
    const size_t& robot_id,
    const lcd::RobotPoseIdSet& vertex_ids) {
  // Create requests msg
  pose_graph_tools_msgs::msg::VLCRequests requests_msg;
  requests_msg.header.stamp = node_->now();
  requests_msg.source_robot_id = config_.my_id_;
  requests_msg.destination_robot_id = robot_id;
  for (const auto& vertex_id : vertex_ids) {
    // Do not request frame that already exists locally
    if (lcd_->frameExists(vertex_id)) {
      continue;
    }
    // Stop if reached batch size
    if (requests_msg.pose_ids.size() >= config_.vlc_batch_size_) {
      break;
    }

    // Double check robot id
    assert(robot_id == vertex_id.first);

    requests_msg.pose_ids.push_back(vertex_id.second);
  }

  vlc_requests_pub_->publish(requests_msg);
  RCLCPP_INFO(node_->get_logger(),
              "Published %lu VLC requests to robot %lu.",
              requests_msg.pose_ids.size(),
              robot_id);
}

bool DistributedLoopClosureRos::requestVLCFrameService(
    const lcd::RobotPoseIdSet& vertex_ids) {
  RCLCPP_WARN(node_->get_logger(),
              "Requesting %zu local VLC frames from Kimera-VIO.",
              vertex_ids.size());

  // Request local VLC frames
  // Populate requested pose ids in ROS service query
  pose_graph_tools_msgs::srv::VLCFrameQuery::Request query;
  std::string service_name =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/vlc_frame_query";
  query.robot_id = config_.my_id_;

  // Populate the pose ids to request
  for (const auto& vertex_id : vertex_ids) {
    // Do not request frame that already exists locally
    if (lcd_->frameExists(vertex_id)) {
      continue;
    }
    // Stop if reaching batch size
    if (query.pose_ids.size() >= static_cast<size_t>(config_.vlc_batch_size_)) {
      break;
    }
    // We can only request via service local frames
    // Frames from other robots have to be requested by publisher
    assert(vertex_id.first == config_.my_id_);
    query.pose_ids.push_back(vertex_id.second);
  }

  // Call ROS2 service
  auto service_client =
      node_->create_client<pose_graph_tools_msgs::srv::VLCFrameQuery>(service_name);

  if (!service_client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
        node_->get_logger(), "ROS service %s does not exist!", service_name.c_str());
    return false;
  }

  auto request = std::make_shared<pose_graph_tools_msgs::srv::VLCFrameQuery::Request>();
  request->robot_id = query.robot_id;
  request->pose_ids = query.pose_ids;

  auto future = service_client->async_send_request(request);

  if (rclcpp::spin_until_future_complete(node_, future) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(), "Could not query VLC frame!");
    return false;
  }

  auto response = future.get();

  // Parse response
  for (const auto& frame_msg : response->frames) {
    lcd::VLCFrame frame;
    kimera_multi_lcd::VLCFrameFromMsg(frame_msg, &frame);
    assert(frame.robot_id_ == my_id_);
    lcd::RobotPoseId vertex_id(frame.robot_id_, frame.pose_id_);
    {  // start lcd critical section
      std::unique_lock<std::mutex> lcd_lock(lcd_mutex_);
      // Fill in submap information for this keyframe
      const auto keyframe = submap_atlas_->getKeyframe(frame.pose_id_);
      if (!keyframe) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
                           "Received VLC frame " << frame.pose_id_
                                                 << " does not exist in submap atlas.");
        continue;
      }
      frame.submap_id_ = CHECK_NOTNULL(keyframe->getSubmap())->id();
      frame.T_submap_pose_ = keyframe->getPoseInSubmapFrame();
      lcd_->addVLCFrame(vertex_id, frame);
    }  // end lcd critical section
  }
  return true;
}

void DistributedLoopClosureRos::vlcResponsesCallback(
    const pose_graph_tools_msgs::msg::VLCFrames::SharedPtr msg) {
  for (const auto& frame_msg : msg->frames) {
    lcd::VLCFrame frame;
    kimera_multi_lcd::VLCFrameFromMsg(frame_msg, &frame);
    lcd::RobotPoseId vertex_id(frame.robot_id_, frame.pose_id_);
    {  // start lcd critical section
      std::unique_lock<std::mutex> lcd_lock(lcd_mutex_);
      lcd_->addVLCFrame(vertex_id, frame);
    }  // end lcd critical section
    // Inter-robot request will be counted as communication
    if (frame.robot_id_ != config_.my_id_) {
      received_vlc_bytes_.push_back(
          kimera_multi_lcd::computeVLCFramePayloadBytes(frame_msg));
    }
    if (config_.run_offline_) {
      offline_robot_pose_msg_[vertex_id] = frame_msg;
    }
  }
  // RCLCPP_INFO(node_->get_logger(),"Received %d VLC frames. ", msg->frames.size());
  if (config_.run_offline_) {
    processOfflineLoopClosures();
  }
}

void DistributedLoopClosureRos::internalVLCCallback(
    const pose_graph_tools_msgs::msg::VLCFrames::SharedPtr msg) {
  processInternalVLC(msg);
}

void DistributedLoopClosureRos::bowRequestsCallback(
    const pose_graph_tools_msgs::msg::BowRequests::SharedPtr msg) {
  if (msg->destination_robot_id != config_.my_id_) return;
  if (msg->source_robot_id == config_.my_id_) {
    RCLCPP_ERROR(node_->get_logger(), "Received BoW requests from myself!");
    return;
  }
  // Push requested Bow Frame IDs to be transmitted later
  std::unique_lock<std::mutex> requested_bows_lock(requested_bows_mutex_);
  if (requested_bows_.find(msg->source_robot_id) == requested_bows_.end())
    requested_bows_[msg->source_robot_id] = std::set<lcd::PoseId>();
  for (const auto& pose_id : msg->pose_ids) {
    requested_bows_[msg->source_robot_id].emplace(pose_id);
  }
}

void DistributedLoopClosureRos::vlcRequestsCallback(
    const pose_graph_tools_msgs::msg::VLCRequests::SharedPtr msg) {
  if (msg->destination_robot_id != config_.my_id_) {
    return;
  }

  if (msg->source_robot_id == config_.my_id_) {
    RCLCPP_ERROR(node_->get_logger(), "Received VLC requests from myself!");
    return;
  }

  if (msg->pose_ids.empty()) {
    return;
  }

  // Find the vlc frames that we are missing
  lcd::RobotPoseIdSet missing_vertex_ids;
  for (const auto& pose_id : msg->pose_ids) {
    lcd::RobotPoseId vertex_id(config_.my_id_, pose_id);
    if (!lcd_->frameExists(vertex_id)) {
      missing_vertex_ids.emplace(vertex_id);
    }
  }

  if (!missing_vertex_ids.empty()) {  // start vlc service critical section
    std::unique_lock<std::mutex> service_lock(vlc_service_mutex_);
    if (!requestVLCFrameService(missing_vertex_ids)) {
      RCLCPP_ERROR(node_->get_logger(),
                   "Failed to retrieve local VLC frames on robot %zu",
                   config_.my_id_);
    }
  }

  // Push requested VLC frame IDs to queue to be transmitted later
  std::unique_lock<std::mutex> requested_frames_lock(requested_frames_mutex_);
  if (requested_frames_.find(msg->source_robot_id) == requested_frames_.end())
    requested_frames_[msg->source_robot_id] = std::set<lcd::PoseId>();
  for (const auto& pose_id : msg->pose_ids) {
    requested_frames_[msg->source_robot_id].emplace(pose_id);
  }
}

void DistributedLoopClosureRos::randomSleep(double min_sec, double max_sec) {
  CHECK(min_sec < max_sec);
  CHECK(min_sec > 0);
  if (max_sec < 1e-3) return;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(min_sec, max_sec);
  double sleep_time = distribution(gen);
  // RCLCPP_INFO(node_->get_logger(),"Sleep %f sec...", sleep_time);
  rclcpp::sleep_for(
      rclcpp::Duration::from_seconds(sleep_time).to_chrono<std::chrono::nanoseconds>());
}

void DistributedLoopClosureRos::publishSubmapOfflineInfo() {
  pose_graph_tools_msgs::msg::VLCFrames msg;
  // Fill in keyframe poses in submaps
  for (int submap_id = 0; submap_id < submap_atlas_->numSubmaps(); ++submap_id) {
    const auto submap = CHECK_NOTNULL(submap_atlas_->getSubmap(submap_id));
    for (const int keyframe_id : submap->getKeyframeIDs()) {
      const auto keyframe = CHECK_NOTNULL(submap->getKeyframe(keyframe_id));
      const auto T_submap_keyframe = keyframe->getPoseInSubmapFrame();
      pose_graph_tools_msgs::msg::VLCFrameMsg frame_msg;
      frame_msg.robot_id = config_.my_id_;
      frame_msg.pose_id = keyframe_id;
      frame_msg.submap_id = submap_id;
      frame_msg.submap_from_pose = GtsamPoseToRos(T_submap_keyframe);
      lcd::RobotPoseId vertex_id(config_.my_id_, keyframe_id);
      offline_robot_pose_msg_[vertex_id] = frame_msg;
      msg.frames.push_back(frame_msg);
    }
  }
  for (lcd::RobotId robot_id = 0; robot_id < config_.my_id_; ++robot_id) {
    msg.destination_robot_id = robot_id;
    vlc_responses_pub_->publish(msg);
    rclcpp::sleep_for(std::chrono::seconds(1));
  }
}

void DistributedLoopClosureRos::save() {
  saveBowVectors(config_.log_output_dir_);
  saveVLCFrames(config_.log_output_dir_);
}
}  // namespace kimera_distributed
