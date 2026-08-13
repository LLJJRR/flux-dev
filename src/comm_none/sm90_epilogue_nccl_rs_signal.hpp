#pragma once

#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/array.h"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "flux/cuda/cuda_common_device.hpp"

namespace cutlass::epilogue::fusion {

template <
    int Stages,
    class TileShape,
    class EpilogueTile,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class SmemLayoutAtom,
    class CopyOpR2S,
    int Alignment = 128 / cute::sizeof_bits_v<Element>>
struct Sm90NcclRsProducerSignal {
  using ElementAux = Element;
  static_assert(Alignment * cute::sizeof_bits_v<Element> % 128 == 0);
  constexpr static bool is_m_major =
      epilogue::collective::detail::is_m_major<StrideMNL>();
  using SmemShapeTma = decltype(cute::make_shape(
      cute::max_common_vector(
          cute::make_layout(cute::get<0>(EpilogueTile{})),
          cute::make_layout(cute::get<0>(EpilogueTile{}))),
      cute::max_common_vector(
          cute::make_layout(cute::get<1>(EpilogueTile{})),
          cute::make_layout(cute::get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(cute::tile_to_shape(
      SmemLayoutAtom{}, SmemShapeTma{},
      cute::conditional_t<is_m_major, cute::Step<cute::_2, cute::_1>,
                          cute::Step<cute::_1, cute::_2>>{}));
  using SmemLayout = decltype(cute::tile_to_shape(
      SmemLayoutTma{},
      cute::make_shape(
          cute::size<0>(cute::shape(EpilogueTile{})),
          cute::size<1>(cute::shape(EpilogueTile{})), cute::Int<Stages>{}),
      cute::conditional_t<is_m_major, cute::Step<cute::_2, cute::_1, cute::_3>,
                          cute::Step<cute::_1, cute::_2, cute::_3>>{}));
  struct Arguments {
    int *producer_ready = nullptr;
    int *tile_counters = nullptr;
    int producer_epoch = 0;
    int world_size = 0;
  };

  using Params = Arguments;
  struct SharedStorage {};

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const &, Arguments const &) { return true; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) { return 0; }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const &, Arguments const &, void *, cudaStream_t,
      CudaHostAdapter * = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE Sm90NcclRsProducerSignal() = default;
  CUTLASS_HOST_DEVICE Sm90NcclRsProducerSignal(Params const &params, SharedStorage const &)
      : params_ptr(&params) {}

  Params const *params_ptr = nullptr;
  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const &) {
    return EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    Params const *params;
    int thread_idx;
    int m;
    int n;
    int tile_m;
    int tile_n;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        Array<ElementAccumulator, FragmentSize> const &, int, int, int,
        Array<ElementInput, FragmentSize> const &input) {
      return input;
    }

    CUTLASS_DEVICE void end() {
      if (params->producer_ready == nullptr)
        return;
      cute::tma_store_wait<0>();
      if (thread_idx != 0)
        return;
      int m_tiles = cute::ceil_div(m, cute::size<0>(TileShape{}));
      int n_tiles = cute::ceil_div(n, cute::size<1>(TileShape{}));
      if (tile_m >= m_tiles || tile_n >= n_tiles)
        return;

      int tiles_m_per_segment = m_tiles / params->world_size;
      int segment = tile_m / tiles_m_per_segment;
      int tiles_per_segment = tiles_m_per_segment * n_tiles;
      bytedance::flux::atomic_ref_sys<int> counter(params->tile_counters[segment]);
      int completed = counter.fetch_add(1, cuda::memory_order_acq_rel) + 1;
      if (completed == tiles_per_segment) {
        bytedance::flux::atomic_store_release_sys(
            params->producer_ready + segment, params->producer_epoch);
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const &args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [tile_m, tile_n, tile_k, tile_l] = args.tile_coord_mnkl;
    return ConsumerStoreCallbacks{
        {}, params_ptr, args.thread_idx, int(M), int(N), int(tile_m), int(tile_n)};
  }
};

}  // namespace cutlass::epilogue::fusion
