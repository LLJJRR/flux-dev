// Copyright 2025 ByteDance Ltd. and/or its affiliates.

#include "gemm_rs/ths_op/moe_gather_rs_nccl.h"

#include "comm_none/ths_op/gemm_grouped_v3.h"
#include "flux/cuda/cuda_common.h"
#include "flux/ths_op/util.h"
#include "gemm_rs/moe_topk_reduce.h"
#include "gemm_rs/ths_op/nccl_signal_reduce_scatter.h"

#include <c10/cuda/CUDAStream.h>

#include <limits>
#include <utility>
#include <vector>

namespace bytedance::flux::ths_op {
namespace {

ncclDataType_t
to_nccl_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ncclFloat16;
    case torch::kBFloat16:
      return ncclBfloat16;
    default:
      FLUX_CHECK(false) << "unsupported NCCL MoE output dtype: " << dtype;
      return ncclNumTypes;
  }
}

}  // namespace

class MoeGatherRSNccl::Impl {
 public:
  Impl(
      std::shared_ptr<Group> group,
      torch::Tensor weight,
      int64_t num_experts,
      int64_t topk,
      int64_t n_split)
      : group_(std::move(group)),
        signal_rs_(group_),
        topk_(topk),
        n_split_(n_split),
        n_dim_(0) {
    FLUX_CHECK_EQ(group_->get_size(), 2)
        << "NCCL MoE GatherRS currently supports TP2 only";
    FLUX_CHECK(get_arch() == ArchEnum::Sm90)
        << "NCCL MoE GatherRS currently supports SM90 only";
    FLUX_CHECK(topk_ == 4 || topk_ == 5)
        << "NCCL MoE GatherRS currently supports topk=4 or topk=5";
    FLUX_CHECK_EQ(weight.dim(), 3);
    n_dim_ = weight.size(1);
    FLUX_CHECK_GT(n_split_, 0);
    FLUX_CHECK_DIV(n_dim_, n_split_);
    int64_t chunk_n = n_dim_ / n_split_;
    FLUX_CHECK_GT(chunk_n, 0);
    FLUX_CHECK_DIV(chunk_n, 256)
        << "each NCCL MoE N split must contain a whole 256-column TopK tile";
    for (int64_t split = 0; split < n_split_; ++split) {
      auto split_weight = weight.narrow(1, split * chunk_n, chunk_n).contiguous();
      grouped_gemms_.emplace_back(
          std::make_unique<GemmGroupedV3>(std::move(split_weight), num_experts));
    }
  }

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      torch::Tensor row_scale) {
    CHECK_INPUT(input, input.scalar_type());
    CHECK_INPUT(routing_idx, torch::kInt32);
    CHECK_INPUT(row_scale, torch::kFloat32);
    FLUX_CHECK_EQ(input.dim(), 2);
    FLUX_CHECK_GT(input.size(0), 0);
    FLUX_CHECK_LE(input.size(0), std::numeric_limits<int>::max());
    FLUX_CHECK_LE(n_dim_, std::numeric_limits<int>::max());
    FLUX_CHECK(
        input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16)
        << "NCCL MoE GatherRS currently supports FP16/BF16 input and weight";
    FLUX_CHECK(!splits_cpu.is_cuda()) << "splits_cpu must reside on CPU";
    FLUX_CHECK_EQ(splits_cpu.sum().item<int64_t>(), input.size(0));
    FLUX_CHECK_EQ(routing_idx.dim(), 1);
    FLUX_CHECK_EQ(row_scale.dim(), 1);
    FLUX_CHECK_EQ(routing_idx.numel(), input.size(0));
    FLUX_CHECK_EQ(row_scale.numel(), input.size(0));
    FLUX_CHECK_DIV(input.size(0), topk_);

    int64_t token_count = input.size(0) / topk_;
    FLUX_CHECK_DIV(token_count, group_->get_size());
    int64_t tokens_per_rank = token_count / group_->get_size();
    ensure_buffers(input, token_count, tokens_per_rank, n_dim_);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (!compute_stream_bound_) {
      compute_stream_ = stream;
      compute_stream_bound_ = true;
    } else {
      FLUX_CHECK_EQ(stream, compute_stream_)
          << "MoeGatherRSNccl reuses internal buffers and must stay on one CUDA stream";
    }

    // Resolve the shared split shape before starting a consumer that can wait
    // indefinitely. Unsupported GEMM configurations then cannot strand NCCL.
    torch::Tensor grouped_output = grouped_gemms_[0]->forward(input, splits_cpu);
    signal_rs_.start_overlap(
        packed_output_.data_ptr(),
        split_output_.data_ptr(),
        output_.numel(),
        to_nccl_dtype(output_.scalar_type()),
        n_split_,
        stream);

    int peer = (group_->get_rank() + 1) % group_->get_size();
    int segment_order[] = {peer, group_->get_rank()};
    int64_t chunk_n = n_dim_ / n_split_;
    // NCCL sees contiguous [rank][N-split][token][N-chunk] blocks. Producing the
    // peer-owned block first follows the TP2 Ring ReduceScatter consumption order.
    try {
      for (int split = 0; split < n_split_; ++split) {
        if (split != 0) {
          grouped_output = grouped_gemms_[split]->forward(input, splits_cpu);
        }
        FLUX_CHECK(
            grouped_output.scalar_type() == torch::kFloat16 ||
            grouped_output.scalar_type() == torch::kBFloat16)
            << "NCCL MoE GatherRS currently supports FP16/BF16 output";
        void *inputs[] = {grouped_output.data_ptr()};
        for (int segment : segment_order) {
          int64_t output_offset =
              (segment * n_split_ + split) * tokens_per_rank * chunk_n;
          moe_topk_reduce_out(
              inputs,
              1,
              from_torch_dtype(grouped_output.scalar_type()),
              routing_idx.data_ptr<int32_t>(),
              topk_,
              static_cast<char *>(packed_output_.data_ptr()) +
                  output_offset * grouped_output.element_size(),
              token_count,
              chunk_n,
              segment * tokens_per_rank,
              tokens_per_rank,
              row_scale.data_ptr<float>(),
              stream);
          signal_rs_.mark_ready(segment, split, stream);
        }
      }
    } catch (...) {
      // Unblock every consumer wait before propagating a host dispatch error.
      for (int split = 0; split < n_split_; ++split) {
        for (int segment = 0; segment < group_->get_size(); ++segment) {
          signal_rs_.mark_ready(segment, split, stream);
        }
      }
      signal_rs_.finish_overlap(stream);
      throw;
    }
    signal_rs_.finish_overlap(stream);
    moe_unpack_split_major_out(
        split_output_.data_ptr(),
        output_.data_ptr(),
        from_torch_dtype(output_.scalar_type()),
        tokens_per_rank,
        n_dim_,
        n_split_,
        stream);
    return output_;
  }

 private:
  void ensure_buffers(
      torch::Tensor const &like, int64_t token_count, int64_t tokens_per_rank, int64_t n_dim) {
    if (!packed_output_.defined() || packed_output_.size(0) != token_count ||
        packed_output_.size(1) != n_dim || packed_output_.scalar_type() != like.scalar_type()) {
      packed_output_ = torch::empty({token_count, n_dim}, like.options());
    }
    if (!output_.defined() || output_.size(0) != tokens_per_rank || output_.size(1) != n_dim ||
        output_.scalar_type() != like.scalar_type()) {
      output_ = torch::empty({tokens_per_rank, n_dim}, like.options());
    }
    if (!split_output_.defined() || split_output_.sizes() != output_.sizes() ||
        split_output_.scalar_type() != like.scalar_type()) {
      split_output_ = torch::empty({tokens_per_rank, n_dim}, like.options());
    }
  }

  std::shared_ptr<Group> group_;
  std::vector<std::unique_ptr<GemmGroupedV3>> grouped_gemms_;
  NcclSignalReduceScatter signal_rs_;
  int64_t topk_;
  int64_t n_split_;
  int64_t n_dim_;
  torch::Tensor packed_output_;
  torch::Tensor split_output_;
  torch::Tensor output_;
  cudaStream_t compute_stream_ = nullptr;
  bool compute_stream_bound_ = false;
};

MoeGatherRSNccl::MoeGatherRSNccl(
    std::shared_ptr<Group> group,
    torch::Tensor weight,
    int64_t num_experts,
    int64_t topk,
    int64_t n_split)
    : impl_(std::make_unique<Impl>(
          std::move(group), std::move(weight), num_experts, topk, n_split)) {}

MoeGatherRSNccl::~MoeGatherRSNccl() = default;

torch::Tensor
MoeGatherRSNccl::forward(
    torch::Tensor input,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    torch::Tensor row_scale) {
  return impl_->forward(
      std::move(input), std::move(splits_cpu), std::move(routing_idx), std::move(row_scale));
}

}  // namespace bytedance::flux::ths_op
