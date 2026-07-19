#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <cuda_runtime_api.h>

namespace bytedance::flux::ths_op {

bool ag_event_profile_enabled();
bool ag_profile_context_active();
const char *ag_profile_mode();
uint64_t ag_profile_launch_id();
void ag_profile_append(std::string text);
void ag_profile_flush();

class AgEventProfilerScope {
 public:
  AgEventProfilerScope(const char *mode, int rank, uint64_t launch_id);
  ~AgEventProfilerScope();
  AgEventProfilerScope(const AgEventProfilerScope &) = delete;
  AgEventProfilerScope &operator=(const AgEventProfilerScope &) = delete;
};

class AgEventTimer {
 public:
  AgEventTimer(const char *name, int rank, cudaStream_t stream);
  ~AgEventTimer();

 private:
  const char *name_;
  const char *mode_;
  int rank_;
  uint64_t launch_id_;
  cudaStream_t stream_;
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
  bool enabled_;
};

}  // namespace bytedance::flux::ths_op
