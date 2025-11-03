#pragma once

#include <aria_viz/visualizer_rerun.h>
#include <glog/logging.h>
#include <gtsam/slam/dataset.h>
#include <spdlog/fmt/fmt.h>

#include <chrono>
#include <future>

namespace kimera_distributed {

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

class RerunVisualizer : public aria::viz::VisualizerRerun {
 public:
  struct Params {
    std::string base_link_frame_id = "baselink";
    std::string odom_frame_id = "odom";
    std::string map_frame_id = "map";
    std::string gt_csv_file = "";
    std::optional<std::string> recording_id = std::nullopt;
    std::string result_dir = "rerun_results";
  };

  RerunVisualizer(const Params& params)
      : RerunVisualizer(params.base_link_frame_id,
                        params.odom_frame_id,
                        params.map_frame_id,
                        params.recording_id) {}

  RerunVisualizer(std::string base_link_frame_id = "baselink",
                  std::string odom_frame_id = "odom",
                  std::string map_frame_id = "map",
                  std::optional<std::string> recording_id = std::nullopt)
      : aria::viz::VisualizerRerun(
            aria::viz::VisualizerRerun::Params("kimera_distributed",
                                               recording_id,
                                               "rerun+http://172.17.0.1:9876/proxy")),
        baselink_(base_link_frame_id),
        map_(map_frame_id),
        odom_(odom_frame_id) {
    // draw the origin frame for visualization
    this->drawTf(map_, Pose3::Identity(), 0.3, true);

    if (not g_custom_sink) {
      AddGlogCustomSink([this](google::LogSeverity severity,
                               const char* filename,
                               int line,
                               const char* message) {
        logGlogMessages(severity, filename, line, message);
      });
      RedirectStdCoutToGlog();
    }
  }

  virtual ~RerunVisualizer() = default;

  // Hold the active custom sink so it persists for the program lifetime.
  static std::unique_ptr<CustomLogSink> g_custom_sink;

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

  void logGlogMessages(google::LogSeverity severity,
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
    this->rec()->log("glog",
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

 private:
  std::filesystem::path baselink_;
  std::filesystem::path map_;
  std::filesystem::path odom_;

  std::mutex rerun_mutex_;
};

}  // namespace kimera_distributed