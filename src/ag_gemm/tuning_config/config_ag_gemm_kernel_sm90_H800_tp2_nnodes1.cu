//===- config_ag_gemm_kernel_sm90_H800_tp2_nnodes1.cu ------------------ C++ ---===//
//
// Temporary H100-as-H800 TP2 tuning config for AGKernel validation.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "flux/op_registry.h"

namespace bytedance::flux {
using namespace cute;

static int config_ag_gemm_kernel_sm90_h800_tp2_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();

  // H=12288, I=49152, TP=2 => local N = 24576
  // M=512
  inst.add(
      make_gemm_meta(
          make_gemm_dtype_config(_BF16{}(), _BF16{}(), _Void{}(), _BF16{}()),
          _Sm90{}(),
          _H800{}(),
          _AGKernel{}(),
          _RRR{}(),
          _GemmV3{}()),
      make_runtime_config(512, 24576, 12288, make_all_gather_runtime_config(2, 1, 0)),
      make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(2l, 1l, 1l)),
          None{},
          cute::make_tuple(256l, 128l, 64l),
          _GemmDefault{}(),
          4,
          _RasterAlongN{}()));

  // M=1024
  inst.add(
      make_gemm_meta(
          make_gemm_dtype_config(_BF16{}(), _BF16{}(), _Void{}(), _BF16{}()),
          _Sm90{}(),
          _H800{}(),
          _AGKernel{}(),
          _RRR{}(),
          _GemmV3{}()),
      make_runtime_config(1024, 24576, 12288, make_all_gather_runtime_config(2, 1, 0)),
      make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(2l, 1l, 1l)),
          None{},
          cute::make_tuple(128l, 256l, 64l),
          _GemmDefault{}(),
          4,
          _RasterAlongM{}()));

  // M=2048
  inst.add(
      make_gemm_meta(
          make_gemm_dtype_config(_BF16{}(), _BF16{}(), _Void{}(), _BF16{}()),
          _Sm90{}(),
          _H800{}(),
          _AGKernel{}(),
          _RRR{}(),
          _GemmV3{}()),
      make_runtime_config(2048, 24576, 12288, make_all_gather_runtime_config(2, 1, 0)),
      make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(2l, 1l, 1l)),
          None{},
          cute::make_tuple(256l, 128l, 64l),
          _GemmDefault{}(),
          4,
          _RasterAlongN{}()));

  // M=4096
  inst.add(
      make_gemm_meta(
          make_gemm_dtype_config(_BF16{}(), _BF16{}(), _Void{}(), _BF16{}()),
          _Sm90{}(),
          _H800{}(),
          _AGKernel{}(),
          _RRR{}(),
          _GemmV3{}()),
      make_runtime_config(4096, 24576, 12288, make_all_gather_runtime_config(2, 1, 0)),
      make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(1l, 2l, 1l)),
          None{},
          cute::make_tuple(128l, 256l, 64l),
          _GemmDefault{}(),
          3,
          _RasterAlongM{}()));

  // M=8192
  inst.add(
      make_gemm_meta(
          make_gemm_dtype_config(_BF16{}(), _BF16{}(), _Void{}(), _BF16{}()),
          _Sm90{}(),
          _H800{}(),
          _AGKernel{}(),
          _RRR{}(),
          _GemmV3{}()),
      make_runtime_config(8192, 24576, 12288, make_all_gather_runtime_config(2, 1, 0)),
      make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(2l, 1l, 1l)),
          None{},
          cute::make_tuple(256l, 128l, 64l),
          _GemmDefault{}(),
          4,
          _RasterAlongN{}()));

  return 0;
}();

}  // namespace bytedance::flux
// clang-format on