#include "kimera_distributed/RerunVisualizer.h"

namespace kimera_distributed {
std::unique_ptr<CustomLogSink> RerunVisualizer::g_custom_sink{nullptr};
std::streambuf* RerunVisualizer::g_original_cout_buf_{nullptr};
GlogStreamBuf RerunVisualizer::g_glog_streambuf_;
}  // namespace kimera_distributed