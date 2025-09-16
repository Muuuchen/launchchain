#ifndef GEMM_HPP
#define GEMM_HPP
#include <cutlass/util/command_line.h>
#include <torch/torch.h>

#include "collective/builder.hpp"
#include "collective/dispatch_policy_extra.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "gemm_cmd.hpp"
#include "helper.hpp"
#include "kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp"
#include "pipeline/prefetch_pipeline_sm90.hpp"
#include "utils.cuh"

void gemm_wrapper(int M, int N, int K, cutlass::float_e4m3_t const *ptrA,
                  cutlass::float_e5m2_t const *ptrB, cutlass::float_e4m3_t *ptrC,
                  cutlass::float_e4m3_t const *ptrD, float overlap_ratio, float prefetch_ratio);
void gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D,
                 float overlap_ratio, float prefetch_ratio);

void gemm_with_prefetch_type_check(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                   torch::Tensor D, float overlap_ratio, float prefetch_ratio);
torch::Tensor gemm(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> out,
                   torch::Tensor D, int overlap_scale, int prefetch_scale);
#endif