#ifndef BINDING_HPP
#define BINDING_HPP

#include <ATen/core/TensorBody.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <launcher.cuh>
#include <rmsnorm.cuh>
#include <utils.cuh>

template <OverlapType type>
void kernel_dispatcher(int overlap_scale, int prefetch_scale, dim3 grid_dim, dim3 block_dim,
                       size_t shm_size, cudaStream_t stream, float4 *output_ptr,
                       const float4 *input_ptr, const float4 *weight_ptr, int m, int n,
                       float epsilon);

torch::Tensor rmsnorm_cpp(torch::Tensor input, torch::Tensor Weight, int overlap_scale,
                          int prefetch_scale, int overlap_type);


torch::Tensor gemm(torch::Tensor A, torch::Tensor B,
                    c10::optional<torch::Tensor> out,
                    torch::Tensor D, int overlap_scale,
                    int prefetch_scale);  
#endif