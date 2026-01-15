#pragma once

#include <aria_viz/visualizer_rerun.h>
#include <glog/logging.h>
#include <gtsam/slam/dataset.h>
#include <log4cxx/appenderskeleton.h>
#include <log4cxx/helpers/stringhelper.h>
#include <log4cxx/spi/loggingevent.h>
#include <spdlog/fmt/fmt.h>

#include <chrono>
#include <future>

#include "kimera_multi_lcd/visualizer.h"

namespace kimera_distributed {

namespace lcd = kimera_multi_lcd;

using RosLogHandler =
    std::function<void(const std::string& msg, log4cxx::LevelPtr level)>;

class CustomRosAppender : public log4cxx::AppenderSkeleton {
 public:
  CustomRosAppender(RosLogHandler handler) : handler_(std::move(handler)) {}

 protected:
  void append(const log4cxx::spi::LoggingEventPtr& event,
              log4cxx::helpers::Pool& pool) override {
    std::string msg = event->getRenderedMessage();
    auto level = event->getLevel();

    handler_(msg, level);
  }

  void close() override {}
  bool requiresLayout() const override { return false; }

  RosLogHandler handler_;
};

// Alias for the custom log handler signature
using GlogHandler = std::function<void(google::LogSeverity severity,
                                       const char* filename,
                                       int line,
                                       const char* message)>;

// Custom sink that forwards glog messages to the user-provided handler
class CustomLogSink : public google::LogSink {
 public:
  explicit CustomLogSink(GlogHandler handler) : handler_(std::move(handler)) {}

  void send(google::LogSeverity severity,
            const char* full_filename,
            const char* base_filename,
            int line,
            const struct tm* tm_time,
            const char* message,
            size_t message_len) override {
    // Build message string and forward to custom handler
    std::string msg(message, message_len);
    handler_(severity, base_filename, line, msg.c_str());
  }

 private:
  GlogHandler handler_;
};

// -----------------------------------------------------------------------------
// Redirect std::cout to glog at INFO level by installing a custom streambuf
// -----------------------------------------------------------------------------
class GlogStreamBuf : public std::streambuf {
 public:
  GlogStreamBuf() { setp(buffer_, buffer_ + sizeof(buffer_) - 1); }

 protected:
  int_type overflow(int_type ch) override {
    if (ch != traits_type::eof()) {
      *pptr() = static_cast<char>(ch);
      pbump(1);
    }
    if (ch == '\n' || pptr() >= epptr()) {
      flushBuffer();
    }
    return ch;
  }

  int sync() override {
    flushBuffer();
    return 0;
  }

 private:
  void flushBuffer() {
    std::ptrdiff_t len = pptr() - pbase();
    if (len <= 0) return;
    std::string msg(pbase(), len);
    LOG(INFO) << msg;
    pbump(-len);
  }

  char buffer_[1024];
};

class RerunVisualizer : public aria::viz::VisualizerRerun,
                        public kimera_multi_lcd::Visualizer {
 public:
  struct Params {
    std::string entity_prefix = "";
    std::string base_link_frame_id = "baselink";
    std::string odom_frame_id = "odom";
    std::string map_frame_id = "map";
    std::string gt_file = "";
    std::optional<std::string> recording_id = std::nullopt;
    std::string result_dir = "rerun_results";
    std::string robot_name = "robot";
  };

  RerunVisualizer(const Params& params)
      : RerunVisualizer(params.entity_prefix,
                        params.base_link_frame_id,
                        params.odom_frame_id,
                        params.map_frame_id,
                        params.robot_name,
                        params.recording_id) {}

  RerunVisualizer(std::string entity_prefix,
                  std::string base_link_frame_id = "baselink",
                  std::string odom_frame_id = "odom",
                  std::string map_frame_id = "map",
                  std::string robot_name = "robot",
                  std::optional<std::string> recording_id = std::nullopt)
      : aria::viz::VisualizerRerun(
            aria::viz::VisualizerRerun::Params("code-slam",
                                               recording_id,
                                               "rerun+http://172.17.0.1:9876/proxy")),
        baselink_(base_link_frame_id),
        map_("/distributed/" + robot_name + "/" + map_frame_id),
        odom_(odom_frame_id),
        robot_name_(robot_name) {
    // draw the origin frame for visualization
    this->drawTf(map_, Pose3::Identity(), 0.3, true);

    if (not g_custom_sink) {
      AddGlogCustomSink([this, entity_prefix](google::LogSeverity severity,
                                              const char* filename,
                                              int line,
                                              const char* message) {
        logGlogMessages(entity_prefix, severity, filename, line, message);
      });
      RedirectStdCoutToGlog();
    }

    ros_appender_ = std::make_shared<CustomRosAppender>(
        [this, entity_prefix](const std::string& msg, log4cxx::LevelPtr level) {
          // Map log4cxx levels to glog severities
          google::LogSeverity severity;
          if (level->isGreaterOrEqual(log4cxx::Level::getFatal())) {
            severity = google::GLOG_FATAL;
          } else if (level->isGreaterOrEqual(log4cxx::Level::getError())) {
            severity = google::GLOG_ERROR;
          } else if (level->isGreaterOrEqual(log4cxx::Level::getWarn())) {
            severity = google::GLOG_WARNING;
          } else {
            severity = google::GLOG_INFO;
          }
          logGlogMessages(entity_prefix, severity, "ROS", 0, msg.c_str());
        });

    // get ROS root logger
    log4cxx::LoggerPtr root_logger = log4cxx::Logger::getRootLogger();
    root_logger->addAppender(ros_appender_.get());
  }

  virtual ~RerunVisualizer() = default;

  // Hold the active custom sink so it persists for the program lifetime.
  static std::unique_ptr<CustomLogSink> g_custom_sink;

  std::shared_ptr<CustomRosAppender> ros_appender_;

  // Call this after google::InitGoogleLogging(), to attach your custom handler
  // in addition to glog's default sinks (stderr and/or log files).
  inline void AddGlogCustomSink(GlogHandler handler) {
    // Remove previous custom sink if installed
    if (g_custom_sink) {
      google::RemoveLogSink(g_custom_sink.get());
      g_custom_sink.reset();
    }

    // Create and install a new sink; default sinks remain active
    g_custom_sink = std::make_unique<CustomLogSink>(std::move(handler));
    google::AddLogSink(g_custom_sink.get());
  }

  // Preserve original buffer so we can restore cout
  static std::streambuf* g_original_cout_buf_;
  static GlogStreamBuf g_glog_streambuf_;

  // Call after InitGoogleLogging() to capture std::cout output
  inline void RedirectStdCoutToGlog() {
    if (!g_original_cout_buf_) {
      g_original_cout_buf_ = std::cout.rdbuf(&g_glog_streambuf_);
    }
  }

  // Restore original std::cout behavior
  inline void RestoreStdCout() {
    if (g_original_cout_buf_) {
      std::cout.rdbuf(g_original_cout_buf_);
      g_original_cout_buf_ = nullptr;
    }
  }

  // Optional: remove the custom sink
  inline void RemoveGlogCustomSink() {
    if (g_custom_sink) {
      google::RemoveLogSink(g_custom_sink.get());
      g_custom_sink.reset();
    }
  }

  void logGlogMessages(std::string log_entity,
                       google::LogSeverity severity,
                       const char* filename,
                       int line,
                       const char* message) {
    // glog severity to Rerun log level
    rerun::TextLogLevel level;
    switch (severity) {
      case google::GLOG_INFO:
        level = rerun::TextLogLevel::Info;
        break;
      case google::GLOG_WARNING:
        level = rerun::TextLogLevel::Warning;
        break;
      case google::GLOG_ERROR:
        level = rerun::TextLogLevel::Error;
        break;
      case google::GLOG_FATAL:
        level = rerun::TextLogLevel::Critical;
        break;
      default:
        level = rerun::TextLogLevel::Debug;  // Default to Debug for other
                                             // severities
    }
    // Forward glog messages to Rerun
    this->rec()->log(log_entity + "/glog",
                     rerun::TextLog(fmt::format("{}", message)).with_level(level));
  }

  void drawCamera(int id,
                  const Pose3& cam_pose,
                  const cv::Mat& image,
                  const cv::Mat& K,
                  bool is_static = true) {
    // draw the tf for this camera
    std::string cam_name = fmt::format("f{}", id);
    this->drawTf(map_ / odom_ / "camera" / cam_name, cam_pose, 0, is_static);

    if (not image.empty()) {
      cv::Mat rgba32;
      if (image.type() == CV_8UC3) {
        cv::cvtColor(image, rgba32, cv::COLOR_BGR2RGBA);
      } else if (image.type() == CV_8UC1) {
        cv::cvtColor(image, rgba32, cv::COLOR_GRAY2RGBA);
      } else if (image.type() == CV_8UC4) {
        rgba32 = image;
      } else {
        throw std::runtime_error("Unsupported image type");
      }
      std::array<float, 9> K_vec = {static_cast<float>(K.at<double>(0, 0)),  // fx
                                    0.f,
                                    static_cast<float>(K.at<double>(0, 2)),  // cx
                                    0.f,
                                    static_cast<float>(K.at<double>(1, 1)),  // fy
                                    static_cast<float>(K.at<double>(1, 2)),  // cy
                                    0.f,
                                    0.f,
                                    1.f};
      rerun::components::PinholeProjection pp(K_vec);

      this->rec()->log_with_static(
          (map_ / odom_ / "camera" / cam_name).c_str(),
          is_static,
          rerun::Image::from_rgba32(
              rgba32,
              {static_cast<uint32_t>(image.cols), static_cast<uint32_t>(image.rows)}));
      // draw the camera
      this->rec()->log_with_static(
          (map_ / odom_ / "camera" / cam_name).c_str(),
          is_static,
          rerun::Pinhole::from_focal_length_and_resolution(
              K_vec[0],
              {static_cast<float>(image.cols), static_cast<float>(image.rows)})
              .with_image_plane_distance(0.1f)
              .with_camera_xyz(rerun::components::ViewCoordinates::FRD));
      // .with_image_from_camera(pp));
    }
  }

  std::vector<Eigen::Vector4f> getColorsFromFactorsType(
      const NonlinearFactorGraph& factors) {
    std::vector<Eigen::Vector4f> colors(factors.size(), aria::viz::ColorMap::kBlack);
    for (size_t i = 0; i < factors.size(); ++i) {
      const auto& factor = factors[i];
      if (factor == nullptr) {
        continue;
      }
      auto keys = factor->keys();
      if (keys.size() > 2) continue;
      uint64_t diff = (keys[0] > keys[1]) ? (keys[0] - keys[1]) : (keys[1] - keys[0]);
      if (diff == 1 or keys.size() == 1) {                // odometry edge
        colors[i] = Eigen::Vector4f(255, 255, 0.0, 255);  // yellow
        continue;
      } else if (diff > 1 and keys.size() == 2) {  // loop closure edge
        auto between_factor =
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
        if (not between_factor) {
          factor->print();
          LOG(FATAL) << "Factor is not a BetweenFactor";
        }
        auto noise = between_factor->noiseModel();
        CHECK(noise);
        auto gauss = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(noise);
        if (not gauss) {
          auto robust = boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(noise);
          gauss =
              boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(robust->noise());
        }
        CHECK(gauss);
        auto info = gauss->information();
        double trans_precision = info.block<3, 3>(3, 3).norm();
        // if trans precision too small, then rot only factor
        if (trans_precision < 1e-6) {
          colors[i] = aria::viz::ColorMap::kRed;
        } else {
          colors[i] = aria::viz::ColorMap::kGreen;
        }
      }
    }
    return colors;
  }

  void visualizeMatchesVersors(lcd::VLCFrame* frame1,
                               lcd::VLCFrame* frame2,
                               const lcd::BearingVectors& versors1,
                               const lcd::BearingVectors& versors2) override {
    std::vector<std::array<float, 2>> vectors;
    std::vector<std::array<float, 2>> origins;
    for (size_t i = 0; i < versors1.size(); ++i) {
      origins.push_back(
          {static_cast<float>(versors1[i].x()), static_cast<float>(versors1[i].y())});
      vectors.push_back({static_cast<float>(versors2[i].x() - versors1[i].x()),
                         static_cast<float>(versors2[i].y() - versors1[i].y())});
    }
    this->rec()->log((robot_name_ + "/versor_arrows").c_str(),
                     rerun::Arrows2D::from_vectors(vectors)
                         .with_origins(origins)
                         .with_radii({rerun::components::Radius::ui_points(1)})
                         .with_colors({rerun::components::Color{255, 0, 0}}));
  }

  /**
   * @brief Visualize matches between keypoints from two frames
   * @param frame1 The first VLCFrame
   * @param frame2 The second VLCFrame
   * @param matches Vector of pairs (index in frame1, index in frame2)
   */
  void visualizeMatchesKeypoints(const std::string& entity,
                                 lcd::VLCFrame* frame1,
                                 lcd::VLCFrame* frame2,
                                 const std::vector<unsigned int>& match1,
                                 const std::vector<unsigned int>& match2) override {
    std::vector<std::array<float, 2>> vectors;
    std::vector<std::array<float, 2>> origins;
    for (size_t i = 0; i < match1.size(); ++i) {
      const auto& kp1 = frame1->keypoints_[match1[i]];
      const auto& kp2 = frame2->keypoints_[match2[i]];
      origins.push_back({kp1.x, kp1.y});
      vectors.push_back({kp2.x - kp1.x, kp2.y - kp1.y});
    }
    this->rec()->log(entity,
                     rerun::Arrows2D::from_vectors(vectors)
                         .with_origins(origins)
                         .with_radii({rerun::components::Radius::ui_points(1)})
                         .with_colors({rerun::components::Color{255, 0, 0}}));
  }

  void loadGTTrajectory(const std::string& gt_file, lcd::RobotId robot_id) {
    std::lock_guard<std::mutex> lock(rerun_mutex_);
    CHECK(!gt_file.empty());

    // load tum txt gt file
    // TUM format: timestamp tx ty tz qx qy qz qw
    std::ifstream file(gt_file);
    CHECK(file.is_open());

    std::string line;
    size_t pose_idx = 0;
    while (std::getline(file, line)) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream iss(line);
      double timestamp, tx, ty, tz, qx, qy, qz, qw;
      if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
        LOG(WARNING) << "Failed to parse line: " << line;
        continue;
      }

      // Create GTSAM Pose3 from TUM format (qx, qy, qz, qw, tx, ty, tz)
      gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
      gtsam::Point3 translation(tx, ty, tz);
      gtsam::Pose3 pose(rotation, translation);

      // Store in trajectory with incrementing index as key
      gtsam::Symbol key(robot_id, pose_idx);  // 'g' for ground truth
      gt_trajectories_.insert(key, pose);
      gt_timestamps_[static_cast<uint64_t>(timestamp * 1e9)] = key;
      pose_idx++;
    }

    file.close();
    LOG(INFO) << "Loaded " << pose_idx << " GT poses from " << gt_file;
  }

  void visualizeGTTrajectories() {
    std::lock_guard<std::mutex> lock(rerun_mutex_);
    std::vector<Point3> points;
    int skip = 10;
    for (const auto& key_value : gt_trajectories_) {
      if (Symbol(key_value.key).index() % skip != 0) {
        continue;
      }
      points.push_back(key_value.value.cast<Pose3>().translation());
    }
    this->drawPoints((robot_name_ + "/gt/trajectories").c_str(),
                     points,
                     aria::viz::ColorMap::kGray,
                     {1.0f},
                     {},
                     true);
  }

  void visualizeCandidates(std::string name,
                           const lcd::RobotPoseId& query_id,
                           const std::vector<lcd::RobotPoseId>& candidate_ids,
                           const std::vector<float>& candidate_scores,
                           const std::map<lcd::RobotPoseId, uint64_t>& timestamps) {
    std::lock_guard<std::mutex> lock(rerun_mutex_);
    if (candidate_ids.empty()) {
      return;
    }

    KeyVector keys;

    // Tolerance for timestamp matching: 50 ms (in nanoseconds)
    const uint64_t kToleranceNs = static_cast<uint64_t>(0.05 * 1e9);

    // Helper lambda to find closest GT timestamp within tolerance
    auto find_closest_gt =
        [this, kToleranceNs](uint64_t query_ts) -> std::optional<gtsam::Symbol> {
      if (gt_timestamps_.empty()) return std::nullopt;

      auto it = gt_timestamps_.lower_bound(query_ts);
      std::optional<gtsam::Symbol> best;
      uint64_t best_diff = std::numeric_limits<uint64_t>::max();

      // Check the iterator at or after query_ts
      if (it != gt_timestamps_.end()) {
        uint64_t diff =
            (it->first > query_ts) ? (it->first - query_ts) : (query_ts - it->first);
        if (diff < best_diff) {
          best_diff = diff;
          best = it->second;
        }
      }

      // Check the previous timestamp (before query_ts)
      if (it != gt_timestamps_.begin()) {
        auto it_prev = std::prev(it);
        uint64_t diff = (it_prev->first > query_ts) ? (it_prev->first - query_ts)
                                                    : (query_ts - it_prev->first);
        if (diff < best_diff) {
          best_diff = diff;
          best = it_prev->second;
        }
      }

      // Return best match only if within tolerance
      if (best && best_diff <= kToleranceNs) return best;
      return std::nullopt;
    };

    auto it_query_ts = timestamps.find(query_id);
    if (it_query_ts == timestamps.end()) {
      LOG(FATAL) << "Query id not found in timestamps map: (" << query_id.first << ", "
                 << query_id.second << ")";
    }
    auto opt_query_sym = find_closest_gt(it_query_ts->second);
    if (not opt_query_sym) {
      LOG(WARNING) << "Could not find GT match for query timestamp: "
                   << it_query_ts->second << ". First gt timestamp: "
                   << (gt_timestamps_.empty() ? 0 : gt_timestamps_.begin()->first);
      return;
    }
    keys.push_back(*opt_query_sym);

    for (const auto& candidate_id : candidate_ids) {
      // find the closest gt timestamp to the candidate timestamp, with some tolerance
      auto it_cand_ts = timestamps.find(candidate_id);

      if (it_cand_ts == timestamps.end()) {
        LOG(FATAL) << "Candidate id not found in timestamps map: ("
                   << candidate_id.first << ", " << candidate_id.second << ")";
      }
      auto opt_cand_sym = find_closest_gt(it_cand_ts->second);

      if (opt_cand_sym && opt_query_sym) {
        keys.push_back(*opt_cand_sym);
      } else {
        LOG(WARNING) << "Could not find GT match for candidate or query timestamp."
                   << " Candidate ts: " << it_cand_ts->second
                   << ", Query ts: " << it_query_ts->second << "first gt timestamp: "
                   << (gt_timestamps_.empty() ? 0 : gt_timestamps_.begin()->first);
      }
    }
    std::vector<gtsam::Point3> gt_points;
    for (const auto& key : keys) {
      gt_points.push_back(gt_trajectories_.at<gtsam::Pose3>(key).translation());
    }
    std::string entry = fmt::format(
        "{}/gt/{}-{}/{}", robot_name_, query_id.first, candidate_ids[0].first, name);
    std::vector<std::string> labels;
    labels.push_back("q");
    for (const auto& score : candidate_scores) {
      labels.push_back(fmt::format("{:.2f}", score));
    }
    this->drawPoints(
        entry, gt_points, aria::viz::ColorMap::kBlue, {2.f}, labels, false);
  }

  void visualizeCandidates(std::string name,
                           const lcd::RobotPoseId& query_id,
                           const std::vector<lcd::RobotPoseId>& candidate_ids,
                           const std::vector<float>& candidate_scores) override {
    visualizeCandidates(
        name, query_id, candidate_ids, candidate_scores, pose_timestamp_map_);
  }

  void updatePoseTimestampMap(const std::map<lcd::RobotPoseId, uint64_t>& new_map) {
    std::lock_guard<std::mutex> lock(rerun_mutex_);
    pose_timestamp_map_ = new_map;
  }

 private:
  std::filesystem::path baselink_;
  std::filesystem::path map_;
  std::filesystem::path odom_;
  std::string robot_name_;

  std::mutex rerun_mutex_;

  gtsam::Values gt_trajectories_;
  std::map<uint64_t, gtsam::Symbol> gt_timestamps_;

  std::map<lcd::RobotPoseId, uint64_t> pose_timestamp_map_;
};

}  // namespace kimera_distributed