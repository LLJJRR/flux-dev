#pragma once

#include <cuda_runtime_api.h>

namespace bytedance::flux::ths_op {

bool ag_event_profile_enabled();

class AgEventTimer {
 public:
  AgEventTimer(const char *name, int rank, cudaStream_t stream);
  ~AgEventTimer();

 private:
  const char *name_;
  int rank_;
  cudaStream_t stream_;
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
  bool enabled_;
};

}  // namespace bytedance::flux::ths_op
