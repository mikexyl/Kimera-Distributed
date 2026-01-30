/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu) Yulun Tian (yulun@mit.edu)
 */

#include "kimera_distributed/DistributedLoopClosureRos.h"

#include <DBoW2/DBoW2.h>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/InitializePose3.h>
#include <kimera_multi_lcd/utils.h>
#include <pose_graph_tools_msgs/PoseGraph.h>
#include <pose_graph_tools_msgs/VLCFrameQuery.h>
#include <pose_graph_tools_ros/utils.h>
#include <ros/console.h>
#include <ros/ros.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "kimera_distributed/Submap.h"
#include "kimera_distributed/configs.h"

namespace kimera_distributed {

DistributedLoopClosureRos::DistributedLoopClosureRos(const ros::NodeHandle& n)
    : nh_(n) {
  DistributedLoopClosureConfig config;
  int my_id_int = -1;
  int num_robots_int = -1;
  ros::param::get("~robot_id", my_id_int);
  ros::param::get("~num_robots", num_robots_int);
  ros::param::get("~frame_id", config.frame_id_);
  assert(my_id_int >= 0);
  assert(num_robots_int > 0);
  config.my_id_ = my_id_int;
  config.num_robots_ = num_robots_int;

  // Path to log outputs
  config.log_output_ = ros::param::get("~log_output_path", config.log_output_dir_);
  ros::param::get("~run_offline", config.run_offline_);

  if (config.run_offline_) {
    if (!ros::param::get("~offline_dir", config.offline_dir_)) {
      ROS_ERROR("Offline directory is missing!");
      ros::shutdown();
    }
  }

  // Visual place recognition params
  ros::param::get("~alpha", config.lcd_params_.alpha_);
  ros::param::get("~dist_local", config.lcd_params_.dist_local_);
  ros::param::get("~max_db_results", config.lcd_params_.max_db_results_);
  ros::param::get("~min_nss_factor", config.lcd_params_.min_nss_factor_);

  // Lcd Third Party Wrapper Params
  ros::param::get("~max_nrFrames_between_islands",
                  config.lcd_params_.lcd_tp_params_.max_nrFrames_between_islands_);
  ros::param::get("~max_nrFrames_between_queries",
                  config.lcd_params_.lcd_tp_params_.max_nrFrames_between_queries_);
  ros::param::get("~max_intraisland_gap",
                  config.lcd_params_.lcd_tp_params_.max_intraisland_gap_);
  ros::param::get("~min_matches_per_island",
                  config.lcd_params_.lcd_tp_params_.min_matches_per_island_);
  ros::param::get("~min_temporal_matches",
                  config.lcd_params_.lcd_tp_params_.min_temporal_matches_);

  // Geometric verification params
  ros::param::get("~ransac_threshold_mono", config.lcd_params_.ransac_threshold_mono_);
  ros::param::get("~ransac_inlier_percentage_mono",
                  config.lcd_params_.ransac_inlier_percentage_mono_);
  ros::param::get("~max_ransac_iterations_mono",
                  config.lcd_params_.max_ransac_iterations_mono_);
  ros::param::get("~lowe_ratio", config.lcd_params_.lowe_ratio_);
  ros::param::get("~max_ransac_iterations", config.lcd_params_.max_ransac_iterations_);
  ros::param::get("~ransac_threshold", config.lcd_params_.ransac_threshold_);
  ros::param::get("~geometric_verification_min_inlier_count",
                  config.lcd_params_.geometric_verification_min_inlier_count_);
  ros::param::get("~geometric_verification_min_inlier_percentage",
                  config.lcd_params_.geometric_verification_min_inlier_percentage_);
  ros::param::get("~avg_focal_length", config.lcd_params_.avg_focal_length_);
  ros::param::get("~detect_interrobot_only", config.lcd_params_.inter_robot_only_);

  ros::param::get("~vocabulary_path", config.lcd_params_.vocab_path_);
  ros::param::get("~lcd_lg_num_features", config.lcd_params_.lcd_lg_num_features_);
  ros::param::get("~lcd_faiss_index_path", config.lcd_params_.lcd_faiss_index_path_);
  ros::param::get("~xfeat_nv_head_model_path",
                  config.lcd_params_.xfeat_nv_head_model_path_);
  ros::param::get("~netvlad_model_path", config.lcd_params_.netvlad_model_path_);
  ros::param::get("~lcd_lg_model_path", config.lcd_params_.lcd_lg_model_path_);
  ros::param::get("~lcd_min_matched_features",
                  config.lcd_params_.lcd_min_matched_features_);

  ros::param::get("~image_width", config.lcd_params_.image_width_);
  ros::param::get("~image_height", config.lcd_params_.image_height_);

  ros::param::get("~network_input_width", config.lcd_params_.network_input_width_);
  ros::param::get("~network_input_height", config.lcd_params_.network_input_height_);

  ros::param::get("~detection_batch_size", config.detection_batch_size_);
  ros::param::get("~bow_skip_num", config.bow_skip_num_);

  CHECK_EQ(config.bow_skip_num_, 1) << "disable bow skip for now";

  // Load parameters controlling VLC communication
  ros::param::get("~bow_batch_size", config.bow_batch_size_);
  ros::param::get("~vlc_batch_size", config.vlc_batch_size_);
  ros::param::get("~loop_batch_size", config.loop_batch_size_);
  ros::param::get("~comm_sleep_time", config.comm_sleep_time_);
  ros::param::get("~loop_sync_sleep_time", config.loop_sync_sleep_time_);

  ros::param::get("~min_sim_vlad", config.lcd_params_.min_sim_vlad);

  // TF
  if (!ros::param::get("~latest_kf_frame_id", latest_kf_frame_id_)) {
    ROS_ERROR("Latest KF frame ID is missing!");
    ros::shutdown();
  }
  if (!ros::param::get("~odom_frame_id", odom_frame_id_)) {
    ROS_ERROR("Odometry frame ID is missing!");
    ros::shutdown();
  }
  if (!ros::param::get("~world_frame_id", world_frame_id_)) {
    ROS_ERROR("World frame ID is missing!");
    ros::shutdown();
  }

  // Load robot names and initialize candidate lc queues
  for (size_t id = 0; id < config.num_robots_; id++) {
    std::string robot_name = "kimera" + std::to_string(id);
    ros::param::get("~robot" + std::to_string(id) + "_name", robot_name);
    config.robot_names_[id] = robot_name;
  }

  CHECK(config.robot_names_.find(config.my_id_) != config.robot_names_.end())
      << "Robot name for robot ID " << config.my_id_ << " not found! have "
      << config.robot_names_.size() << " names loaded.";

  // get time str HHMM
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm = *std::localtime(&now_c);
  char time_str[5];
  std::strftime(time_str, sizeof(time_str), "%H%M", &now_tm);

  std::vector<std::string> gt_files;
  ros::param::get("~gt_files", gt_files);

  rerun_visualizer_ = std::make_shared<RerunVisualizer>(
      "distributed/" + config.robot_names_.at(config.my_id_),
      config.robot_names_.at(config.my_id_),
      odom_frame_id_,
      world_frame_id_,
      config.robot_names_.at(config.my_id_),
      time_str);

  for (int i = 0; i < gt_files.size(); i++) {
    rerun_visualizer_->loadGTTrajectory(gt_files[i], i);
  }

  rerun_visualizer_->visualizeGTTrajectories();

  // std::string(time_str));

  lcd_->setVisualizer(rerun_visualizer_);

  ros::param::get("~max_submap_size", config.submap_params_.max_submap_size);
  ros::param::get("~max_submap_distance", config.submap_params_.max_submap_distance);

  initialize(config);

  // Subscriber
  std::string topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/pose_graph";
  local_pg_sub_ = nh_.subscribe(
      topic, 1000, &DistributedLoopClosureRos::localPoseGraphCallback, this);

  std::string internal_vlc_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/vlc_frames";
  internal_vlc_sub_ = nh_.subscribe(
      internal_vlc_topic, 1000, &DistributedLoopClosureRos::internalVLCCallback, this);

  std::string connectivity_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/connected_peer_ids";
  connectivity_sub_ = nh_.subscribe(
      connectivity_topic, 5, &DistributedLoopClosureRos::connectivityCallback, this);

  for (size_t id = 0; id < config_.num_robots_; ++id) {
    if (id < config_.my_id_) {
      std::string vlc_req_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/vlc_requests";
      ros::Subscriber vlc_req_sub = nh_.subscribe(
          vlc_req_topic, 1, &DistributedLoopClosureRos::vlcRequestsCallback, this);
      vlc_requests_sub_.push_back(vlc_req_sub);

      std::string bow_req_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/bow_requests";
      ros::Subscriber bow_req_sub = nh_.subscribe(
          bow_req_topic, 1, &DistributedLoopClosureRos::bowRequestsCallback, this);
      bow_requests_sub_.push_back(bow_req_sub);
    }

    if (id >= config_.my_id_) {
      std::string bow_topic =
          "/" + config_.robot_names_[id] + "/kimera_vio_ros/bow_query";
      ros::Subscriber bow_sub =
          nh_.subscribe(bow_topic, 1000, &DistributedLoopClosureRos::bowCallback, this);
      bow_sub_.push_back(bow_sub);
      bow_latest_[id] = 0;
      bow_received_[id] = std::unordered_set<lcd::PoseId>();
    }

    if (id > config_.my_id_) {
      std::string resp_topic =
          "/" + config_.robot_names_[id] + "/kimera_distributed/vlc_responses";
      ros::Subscriber resp_sub = nh_.subscribe(
          resp_topic, 10, &DistributedLoopClosureRos::vlcResponsesCallback, this);
      vlc_responses_sub_.push_back(resp_sub);
    }
  }

  // Publisher
  std::string bow_response_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/bow_query";
  bow_response_pub_ =
      nh_.advertise<pose_graph_tools_msgs::BowQueries>(bow_response_topic, 1000, true);

  std::string bow_request_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/bow_requests";
  bow_requests_pub_ =
      nh_.advertise<pose_graph_tools_msgs::BowRequests>(bow_request_topic, 100, true);

  std::string pose_graph_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/pose_graph";
  pose_graph_pub_ =
      nh_.advertise<pose_graph_tools_msgs::PoseGraph>(pose_graph_topic, 1000, true);

  std::string resp_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/vlc_responses";
  vlc_responses_pub_ =
      nh_.advertise<pose_graph_tools_msgs::VLCFrames>(resp_topic, 10, true);

  std::string req_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/vlc_requests";
  vlc_requests_pub_ =
      nh_.advertise<pose_graph_tools_msgs::VLCRequests>(req_topic, 10, true);

  std::string optimized_nodes_topic = "/" + config_.robot_names_[config_.my_id_] +
                                      "/kimera_distributed/optimized_nodes";
  optimized_nodes_pub_ =
      nh_.advertise<pose_graph_tools_msgs::PoseGraph>(optimized_nodes_topic, 1, true);
  std::string optimized_path_topic =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_distributed/optimized_path";
  optimized_path_pub_ = nh_.advertise<nav_msgs::Path>(optimized_path_topic, 1, true);

  if (config_.my_id_ == 0) {
    std::string pose_corrector_topic = "/" + config_.robot_names_[config_.my_id_] +
                                       "/kimera_distributed/pose_corrector";
    dpgo_frame_corrector_pub_ =
        nh_.advertise<geometry_msgs::Pose>(pose_corrector_topic, 1, true);
  }

  log_timer_ = nh_.createTimer(
      ros::Duration(10.0), &DistributedLoopClosureRos::logTimerCallback, this);

  tf_timer_ = nh_.createTimer(
      ros::Duration(1.0), &DistributedLoopClosureRos::tfTimerCallback, this);

  pose_graph_pub_timer_ = nh_.createTimer(
      ros::Duration(1.0), &DistributedLoopClosureRos::poseGraphPubTimerCallback, this);

  LOG(INFO) << "Distributed Kimera node initialized (ID = " << config_.my_id_ << "). \n"
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
            << "interrobot loop closure only = "
            << config_.lcd_params_.inter_robot_only_ << "\n"
            << "maximum batch size to request BoW vectors = " << config_.bow_batch_size_
            << "\n"
            << "maximum batch size to request VLC frames = " << config_.vlc_batch_size_
            << "\n"
            << "Communication thread sleep time = " << config_.comm_sleep_time_ << "\n"
            << "maximum submap size = " << config_.submap_params_.max_submap_size
            << "\n"
            << "maximum submap distance = "
            << config_.submap_params_.max_submap_distance << "\n"
            << "loop detection batch size = " << config_.detection_batch_size_ << "\n"
            << "loop synchronization batch size = " << config_.loop_batch_size_ << "\n"
            << "loop synchronization sleep time = " << config_.loop_sync_sleep_time_
            << "\n"
            << "BoW vector skip num = " << config_.bow_skip_num_ << "\n";

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
  ROS_INFO("Robot %zu started loop detection / place recognition thread.",
           config_.my_id_);

  // Start verification thread
  verification_thread_.reset(
      new std::thread(&DistributedLoopClosureRos::runVerification, this));
  ROS_INFO("Robot %zu started loop verification thread.", config_.my_id_);

  // Start comms thread
  comms_thread_.reset(new std::thread(&DistributedLoopClosureRos::runComms, this));
  ROS_INFO("Robot %zu started communication thread.", config_.my_id_);

  start_time_ = ros::Time::now();
  next_latest_bow_pub_time_ = ros::Time::now();
}

DistributedLoopClosureRos::~DistributedLoopClosureRos() {
  if (config_.log_output_) {
    save();
  }
  ROS_INFO("Shutting down DistributedLoopClosureRos process on robot %zu...",
           config_.my_id_);
}

void DistributedLoopClosureRos::bowCallback(
    const pose_graph_tools_msgs::BowQueriesConstPtr& query_msg) {
  processBow(query_msg);
}

void DistributedLoopClosureRos::localPoseGraphCallback(
    const pose_graph_tools_msgs::PoseGraph::ConstPtr& msg) {
  processLocalPoseGraph(msg);
}

void DistributedLoopClosureRos::poseGraphPubTimerCallback(
    const ros::TimerEvent& event) {
  updateSubmapPoses();
  updateSubmapLoops();

  // Publish sparsified pose graph
  pose_graph_tools_msgs::PoseGraph sparse_pose_graph = getSubmapPoseGraph(false);
  if (!sparse_pose_graph.edges.empty() || !sparse_pose_graph.nodes.empty()) {
    pose_graph_pub_.publish(sparse_pose_graph);
  }
}

void DistributedLoopClosureRos::connectivityCallback(
    const std_msgs::UInt16MultiArrayConstPtr& msg) {
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

void DistributedLoopClosureRos::logTimerCallback(const ros::TimerEvent& event) {
  if (!config_.log_output_) return;
  if (config_.run_offline_) return;
  logLcdStat();
  // Save latest submap atlas
  saveSubmapAtlas(config_.log_output_dir_);
  // Save latest trajectory estimates in the world frame
  if (backend_update_count_ > 0) {
    auto elapsed_time = ros::Time::now() - start_time_;
    int elapsed_sec = int(elapsed_time.toSec());
    std::string file_path = config_.log_output_dir_ + "kimera_distributed_poses_" +
                            std::to_string(elapsed_sec) + ".csv";
    gtsam::Values::shared_ptr nodes_ptr(new gtsam::Values);
    computePosesInWorldFrame(nodes_ptr);
    savePosesToFile(file_path, *nodes_ptr);
  }
}

void DistributedLoopClosureRos::tfTimerCallback(const ros::TimerEvent& event) {
  size_t total_bow_mb =
      std::accumulate(received_bow_bytes_.begin(), received_bow_bytes_.end(), 0);
  rerun_visualizer_->drawScalar(
      config_.robot_names_.at(config_.my_id_) + "/received_bow_byte", total_bow_mb);

  size_t total_vlc_mb =
      std::accumulate(received_vlc_bytes_.begin(), received_vlc_bytes_.end(), 0);
  rerun_visualizer_->drawScalar(
      config_.robot_names_.at(config_.my_id_) + "/received_vlc_byte", total_vlc_mb);
}

void DistributedLoopClosureRos::runDetection() {
  while (ros::ok() && !should_shutdown_) {
    if (!bow_msgs_.empty()) {
      detectLoopSpin();
    }
    ros::Duration(1.0).sleep();
  }
}

void DistributedLoopClosureRos::runVerification() {
  ros::WallRate r(1);
  while (ros::ok() && !should_shutdown_) {
    if (queued_lc_.empty()) {
      r.sleep();
    } else {
      verifyLoopSpin();
    }
  }
}

void DistributedLoopClosureRos::runComms() {
  while (ros::ok() && !should_shutdown_) {
    LOG(INFO) << "Communication thread awake.";
    // Request missing Bow vectors from other robots
    requestBowVectors();
    LOG(INFO) << "Requested missing BoW vectors.";

    // Request VLC frames from other robots
    size_t total_candidates = updateCandidateList();
    LOG(INFO) << "Total VLC frame requests to make: " << total_candidates;
    if (total_candidates > 0) {
      requestFrames();
    }

    // Publish BoW vectors requested by other robots
    publishBowVectors();

    // Publish VLC frames requested by other robots
    publishFrames();

    // Once a while publish latest BoW vector
    // This is needed for other robots to request potentially missing BoW
    if (ros::Time::now().toSec() > next_latest_bow_pub_time_.toSec()) {
      publishLatestBowVector();
      next_latest_bow_pub_time_ += ros::Duration(30);  // TODO: make into parameter
    }

    // Print stats
    ROS_INFO_STREAM("Total inter-robot loop closures: " << num_inter_robot_loops_);

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
  pose_graph_tools_msgs::BowRequests msg;
  msg.source_robot_id = config_.my_id_;
  msg.destination_robot_id = robot_id_to_query;
  for (const auto& pose_id : missing_bow_vectors) {
    if (msg.pose_ids.size() >= config_.bow_batch_size_) break;
    msg.pose_ids.push_back(pose_id);
  }
  ROS_WARN("Processing %lu BoW requests to robot %lu.",
           msg.pose_ids.size(),
           robot_id_to_query);
  bow_requests_pub_.publish(msg);
}

void DistributedLoopClosureRos::publishBowVectors() {
  lcd::RobotId robot_id_to_publish;
  lcd::RobotPoseIdSet bow_vectors_to_publish;
  if (!queryBowVectorsPublish(robot_id_to_publish, bow_vectors_to_publish)) {
    return;
  }
  // Send BoW vectors to selected robot
  pose_graph_tools_msgs::BowQueries msg;
  msg.header.stamp = ros::Time::now();
  msg.destination_robot_id = robot_id_to_publish;
  for (const auto& robot_pose_id : bow_vectors_to_publish) {
    pose_graph_tools_msgs::BowQuery query_msg;
    query_msg.robot_id = robot_pose_id.first;
    query_msg.pose_id = robot_pose_id.second;
    CHECK(pose_timestamp_map_.find(robot_pose_id) != pose_timestamp_map_.end());
    CHECK(pose_timestamp_map_.at(robot_pose_id) != 0);
    query_msg.header.stamp =
        ros::Time().fromNSec(pose_timestamp_map_.at(robot_pose_id));
    kimera_multi_lcd::MatToBowVectorMsg(lcd_->getGlobalDesc(robot_pose_id),
                                        &(query_msg.bow_vector));
    msg.queries.push_back(query_msg);
  }
  bow_response_pub_.publish(msg);
  ROS_INFO("Published %zu BoWs to robot %zu (%zu waiting).",
           msg.queries.size(),
           robot_id_to_publish,
           requested_bows_[robot_id_to_publish].size());
}

void DistributedLoopClosureRos::publishLatestBowVector() {
  int pose_id = lcd_->latestPoseIdWithGlobalDesc(config_.my_id_);
  if (pose_id != -1) {
    lcd::RobotPoseId latest_id(config_.my_id_, pose_id);
    pose_graph_tools_msgs::BowQuery query_msg;
    query_msg.robot_id = config_.my_id_;
    query_msg.pose_id = pose_id;
    CHECK(pose_timestamp_map_.find(latest_id) != pose_timestamp_map_.end());
    CHECK(pose_timestamp_map_.at(latest_id) != 0);
    query_msg.header.stamp = ros::Time().fromNSec(pose_timestamp_map_.at(latest_id));
    kimera_multi_lcd::MatToBowVectorMsg(lcd_->getGlobalDesc(latest_id),
                                        &(query_msg.bow_vector));

    pose_graph_tools_msgs::BowQueries msg;
    msg.header.stamp = ros::Time::now();
    msg.queries.push_back(query_msg);
    for (lcd::RobotId robot_id = 0; robot_id < config_.my_id_; ++robot_id) {
      msg.destination_robot_id = robot_id;
      bow_response_pub_.publish(msg);
    }
  }
  ROS_INFO("Published latest BoW vector.");
}

void DistributedLoopClosureRos::requestFrames() {
  lcd::RobotPoseIdSet my_vertex_ids, target_vertex_ids;
  lcd::RobotId target_robot_id;
  queryFramesRequest(my_vertex_ids, target_robot_id, target_vertex_ids);

  LOG(INFO) << "Requesting VLC frames from self " << my_vertex_ids.size()
            << " and from robot " << target_robot_id << " " << target_vertex_ids.size();

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
    pose_graph_tools_msgs::VLCFrames frames_msg;
    frames_msg.destination_robot_id = target_robot_id;
    for (const auto& vertex_id : target_vertex_ids) {
      pose_graph_tools_msgs::VLCFrameMsg vlc_msg;
      kimera_multi_lcd::VLCFrameToMsg(lcd_->getVLCFrame(vertex_id), &vlc_msg);
      frames_msg.frames.push_back(vlc_msg);
    }
    vlc_responses_pub_.publish(frames_msg);
    ROS_INFO("Published %zu frames to robot %zu (%zu frames waiting).",
             frames_msg.frames.size(),
             target_robot_id,
             requested_frames_[target_robot_id].size());
  }
}

void DistributedLoopClosureRos::processVLCRequests(
    const size_t& robot_id,
    const lcd::RobotPoseIdSet& vertex_ids) {
  if (vertex_ids.size() == 0) {
    return;
  }

  // ROS_INFO("Processing %lu VLC requests to robot %lu.", vertex_ids.size(), robot_id);
  if (robot_id == config_.my_id_) {
    // Directly request from Kimera-VIO-ROS
    {  // start vlc service critical section
      std::unique_lock<std::mutex> service_lock(vlc_service_mutex_);
      if (!requestVLCFrameService(vertex_ids)) {
        ROS_ERROR("Failed to retrieve local VLC frames on robot %zu.", config_.my_id_);
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
  pose_graph_tools_msgs::VLCRequests requests_msg;
  requests_msg.header.stamp = ros::Time::now();
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

  vlc_requests_pub_.publish(requests_msg);
  ROS_INFO("Published %lu VLC requests to robot %lu.",
           requests_msg.pose_ids.size(),
           robot_id);
}

bool DistributedLoopClosureRos::requestVLCFrameService(
    const lcd::RobotPoseIdSet& vertex_ids) {
  ROS_WARN("Requesting %zu local VLC frames from Kimera-VIO.", vertex_ids.size());

  // Request local VLC frames
  // Populate requested pose ids in ROS service query
  pose_graph_tools_msgs::VLCFrameQuery query;
  std::string service_name =
      "/" + config_.robot_names_[config_.my_id_] + "/kimera_vio_ros/vlc_frame_query";
  query.request.robot_id = config_.my_id_;

  // Populate the pose ids to request
  for (const auto& vertex_id : vertex_ids) {
    // Do not request frame that already exists locally
    if (lcd_->frameExists(vertex_id)) {
      continue;
    }
    // Stop if reaching batch size
    if (query.request.pose_ids.size() >= config_.vlc_batch_size_) {
      break;
    }
    // We can only request via service local frames
    // Frames from other robots have to be requested by publisher
    assert(vertex_id.first == config_.my_id_);
    query.request.pose_ids.push_back(vertex_id.second);
  }

  // Call ROS service
  if (!ros::service::waitForService(service_name, ros::Duration(5.0))) {
    ROS_ERROR_STREAM("ROS service " << service_name << " does not exist!");
    return false;
  }
  if (!ros::service::call(service_name, query)) {
    ROS_ERROR_STREAM("Could not query VLC frame!");
    return false;
  }

  // Parse response
  size_t n_accepted_frames{0};
  for (const auto& frame_msg : query.response.frames) {
    lcd::VLCFrame frame;
    kimera_multi_lcd::VLCFrameFromMsg(frame_msg, &frame);
    assert(frame.robot_id_ == my_id_);
    lcd::RobotPoseId vertex_id(frame.robot_id_, frame.pose_id_);
    {  // start lcd critical section
      std::unique_lock<std::mutex> lcd_lock(lcd_mutex_);
      // Fill in submap information for this keyframe
      const auto keyframe = submap_atlas_->getKeyframe(frame.pose_id_);
      if (!keyframe) {
        ROS_WARN_STREAM("Received VLC frame " << frame.pose_id_
                                              << " does not exist in submap atlas.");
        continue;
      }
      frame.submap_id_ = CHECK_NOTNULL(keyframe->getSubmap())->id();
      frame.T_submap_pose_ = keyframe->getPoseInSubmapFrame();
      lcd_->addVLCFrame(vertex_id, frame);
      n_accepted_frames++;
    }  // end lcd critical section
  }
  LOG(INFO) << "accepted " << n_accepted_frames << "/" << query.response.frames.size()
            << " frames";
  return true;
}

void DistributedLoopClosureRos::vlcResponsesCallback(
    const pose_graph_tools_msgs::VLCFramesConstPtr& msg) {
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
  // ROS_INFO("Received %d VLC frames. ", msg->frames.size());
  if (config_.run_offline_) {
    processOfflineLoopClosures();
  }
}

void DistributedLoopClosureRos::internalVLCCallback(
    const pose_graph_tools_msgs::VLCFramesConstPtr& msg) {
  processInternalVLC(msg);
}

void DistributedLoopClosureRos::bowRequestsCallback(
    const pose_graph_tools_msgs::BowRequestsConstPtr& msg) {
  if (msg->destination_robot_id != config_.my_id_) return;
  if (msg->source_robot_id == config_.my_id_) {
    ROS_ERROR("Received BoW requests from myself!");
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
    const pose_graph_tools_msgs::VLCRequestsConstPtr& msg) {
  if (msg->destination_robot_id != config_.my_id_) {
    return;
  }

  if (msg->source_robot_id == config_.my_id_) {
    ROS_ERROR("Received VLC requests from myself!");
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
      ROS_ERROR_STREAM("Failed to retrieve local VLC frames on robot "
                       << config_.my_id_);
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
  // ROS_INFO("Sleep %f sec...", sleep_time);
  ros::Duration(sleep_time).sleep();
}

void DistributedLoopClosureRos::publishSubmapOfflineInfo() {
  pose_graph_tools_msgs::VLCFrames msg;
  // Fill in keyframe poses in submaps
  for (int submap_id = 0; submap_id < submap_atlas_->numSubmaps(); ++submap_id) {
    const auto submap = CHECK_NOTNULL(submap_atlas_->getSubmap(submap_id));
    for (const int keyframe_id : submap->getKeyframeIDs()) {
      const auto keyframe = CHECK_NOTNULL(submap->getKeyframe(keyframe_id));
      const auto T_submap_keyframe = keyframe->getPoseInSubmapFrame();
      pose_graph_tools_msgs::VLCFrameMsg frame_msg;
      frame_msg.robot_id = config_.my_id_;
      frame_msg.pose_id = keyframe_id;
      frame_msg.submap_id = submap_id;
      frame_msg.T_submap_pose = GtsamPoseToRos(T_submap_keyframe);
      lcd::RobotPoseId vertex_id(config_.my_id_, keyframe_id);
      offline_robot_pose_msg_[vertex_id] = frame_msg;
      msg.frames.push_back(frame_msg);
    }
  }
  for (lcd::RobotId robot_id = 0; robot_id < config_.my_id_; ++robot_id) {
    msg.destination_robot_id = robot_id;
    vlc_responses_pub_.publish(msg);
    ros::Duration(1).sleep();
  }
}

void DistributedLoopClosureRos::save() {
  saveBowVectors(config_.log_output_dir_);
  saveVLCFrames(config_.log_output_dir_);
}
}  // namespace kimera_distributed
