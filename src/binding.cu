#include <ATen/core/TensorBody.h>

// #include <pybind11/pybind11.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
// #include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cassert>
#include <cstddef>
#include <launcher.cuh>
#include <rmsnorm.cuh>
#include <utils.cuh>

#include "binding.hpp"
#include "gemm.cuh"

// in cuda impl is fp16
torch::Tensor rmsnorm_cpp(torch::Tensor input, torch::Tensor weight, int overlap_scale,
                          int prefetch_scale, int overlap_type) {
    torch::Tensor output = torch::empty_like(input);
    // typecheck
    if (input.dtype() != torch::kFloat16 || weight.dtype() != torch::kFloat16 ||
        output.dtype() != torch::kFloat16) {
        throw std::runtime_error("input or output or weight's dtype != float16;");
    }
    const int m = input.sizes()[0];
    const int n = input.sizes()[1];
    float4 const *input_ptr = reinterpret_cast<float4 const *>(input.data_ptr<at::Half>());
    float4 const *weight_ptr = reinterpret_cast<float4 const *>(weight.data_ptr<at::Half>());
    float4 *output_ptr = reinterpret_cast<float4 *>(output.data_ptr<at::Half>());
    cudaStream_t stream = 0;
    size_t shm_size = 0;
    dim3 grid_dim(m);
    constexpr float epsilon = 1e-5;
    dim3 block_dim((n <= 4096) ? 128 : 256);

    if (overlap_type == 0) {
        shm_size = 0;
        rmsnorm_twoPassAlgo_e8<<<grid_dim, block_dim, shm_size, stream>>>(
            output_ptr, input_ptr, weight_ptr, m, n, epsilon);
    } else if (overlap_type == 1) {
        kernel_dispatcher<OverlapType::prefetch>(overlap_scale, prefetch_scale, grid_dim, block_dim,
                                                 shm_size, stream, output_ptr, input_ptr,
                                                 weight_ptr, m, n, epsilon);
    } else if (overlap_type == 2) {
        kernel_dispatcher<OverlapType::shared>(overlap_scale, prefetch_scale, grid_dim, block_dim,
                                               shm_size, stream, output_ptr, input_ptr, weight_ptr,
                                               m, n, epsilon);
    }

    return output;
}

//
template <int overlap_scale, int prefetch_scale, OverlapType overlap_type>
void launch_kernel_wrapper(dim3 grid_dim, dim3 block_dim, size_t shm_size, cudaStream_t stream,
                           float4 *output_ptr, const float4 *input_ptr, const float4 *weight_ptr,
                           int m, int n, float epsilon) {
    if constexpr (overlap_type == OverlapType::prefetch) {
        shm_size = 0;
        LAUNCH_KERNEL_WITH_PDL(
            (rmsnorm_twoPassAlgo_e8_prefetch<overlap_scale, prefetch_scale, OverlapType::prefetch>),
            grid_dim, block_dim, shm_size, stream, output_ptr, input_ptr, weight_ptr, m, n,
            epsilon);
    } else if constexpr (overlap_type == OverlapType::shared) {
        shm_size = (2 * n * prefetch_scale / 10) & (~31);  // floor
        LAUNCH_KERNEL_WITH_PDL(
            (rmsnorm_twoPassAlgo_e8_prefetch<overlap_scale, prefetch_scale, OverlapType::shared>),
            grid_dim, block_dim, shm_size, stream, output_ptr, input_ptr, weight_ptr, m, n,
            epsilon);
    }
}

template <OverlapType type>
void kernel_dispatcher(int overlap_scale, int prefetch_scale, dim3 grid_dim, dim3 block_dim,
                       size_t shm_size, cudaStream_t stream, float4 *output_ptr,
                       const float4 *input_ptr, const float4 *weight_ptr, int m, int n,
                       float epsilon) {
    int kernel_id = overlap_scale * 11 + prefetch_scale;

// 使用显式的模板参数定义每个case
#define CASE(id, os, ps)                                                                       \
    case id:                                                                                   \
        launch_kernel_wrapper<os, ps, type>(grid_dim, block_dim, shm_size, stream, output_ptr, \
                                            input_ptr, weight_ptr, m, n, epsilon);             \
        break;

    switch (kernel_id) {
        // overlap_scale = 0
        CASE(0, 0, 0)
        CASE(1, 0, 1)
        CASE(2, 0, 2)
        CASE(3, 0, 3)
        CASE(4, 0, 4)
        CASE(5, 0, 5)
        CASE(6, 0, 6)
        CASE(7, 0, 7)
        CASE(8, 0, 8)
        CASE(9, 0, 9)
        CASE(10, 0, 10)

        // overlap_scale = 1
        CASE(11, 1, 0)
        CASE(12, 1, 1)
        CASE(13, 1, 2)
        CASE(14, 1, 3)
        CASE(15, 1, 4)
        CASE(16, 1, 5)
        CASE(17, 1, 6)
        CASE(18, 1, 7)
        CASE(19, 1, 8)
        CASE(20, 1, 9)
        CASE(21, 1, 10)

        // overlap_scale = 2
        CASE(22, 2, 0)
        CASE(23, 2, 1)
        CASE(24, 2, 2)
        CASE(25, 2, 3)
        CASE(26, 2, 4)
        CASE(27, 2, 5)
        CASE(28, 2, 6)
        CASE(29, 2, 7)
        CASE(30, 2, 8)
        CASE(31, 2, 9)
        CASE(32, 2, 10)

        // overlap_scale = 3
        CASE(33, 3, 0)
        CASE(34, 3, 1)
        CASE(35, 3, 2)
        CASE(36, 3, 3)
        CASE(37, 3, 4)
        CASE(38, 3, 5)
        CASE(39, 3, 6)
        CASE(40, 3, 7)
        CASE(41, 3, 8)
        CASE(42, 3, 9)
        CASE(43, 3, 10)

        // overlap_scale = 4
        CASE(44, 4, 0)
        CASE(45, 4, 1)
        CASE(46, 4, 2)
        CASE(47, 4, 3)
        CASE(48, 4, 4)
        CASE(49, 4, 5)
        CASE(50, 4, 6)
        CASE(51, 4, 7)
        CASE(52, 4, 8)
        CASE(53, 4, 9)
        CASE(54, 4, 10)

        // overlap_scale = 5
        CASE(55, 5, 0)
        CASE(56, 5, 1)
        CASE(57, 5, 2)
        CASE(58, 5, 3)
        CASE(59, 5, 4)
        CASE(60, 5, 5)
        CASE(61, 5, 6)
        CASE(62, 5, 7)
        CASE(63, 5, 8)
        CASE(64, 5, 9)
        CASE(65, 5, 10)

        // overlap_scale = 6
        CASE(66, 6, 0)
        CASE(67, 6, 1)
        CASE(68, 6, 2)
        CASE(69, 6, 3)
        CASE(70, 6, 4)
        CASE(71, 6, 5)
        CASE(72, 6, 6)
        CASE(73, 6, 7)
        CASE(74, 6, 8)
        CASE(75, 6, 9)
        CASE(76, 6, 10)

        // overlap_scale = 7
        CASE(77, 7, 0)
        CASE(78, 7, 1)
        CASE(79, 7, 2)
        CASE(80, 7, 3)
        CASE(81, 7, 4)
        CASE(82, 7, 5)
        CASE(83, 7, 6)
        CASE(84, 7, 7)
        CASE(85, 7, 8)
        CASE(86, 7, 9)
        CASE(87, 7, 10)

        // overlap_scale = 8
        CASE(88, 8, 0)
        CASE(89, 8, 1)
        CASE(90, 8, 2)
        CASE(91, 8, 3)
        CASE(92, 8, 4)
        CASE(93, 8, 5)
        CASE(94, 8, 6)
        CASE(95, 8, 7)
        CASE(96, 8, 8)
        CASE(97, 8, 9)
        CASE(98, 8, 10)

        // overlap_scale = 9
        CASE(99, 9, 0)
        CASE(100, 9, 1)
        CASE(101, 9, 2)
        CASE(102, 9, 3)
        CASE(103, 9, 4)
        CASE(104, 9, 5)
        CASE(105, 9, 6)
        CASE(106, 9, 7)
        CASE(107, 9, 8)
        CASE(108, 9, 9)
        CASE(109, 9, 10)

        // overlap_scale = 10
        CASE(110, 10, 0)
        CASE(111, 10, 1)
        CASE(112, 10, 2)
        CASE(113, 10, 3)
        CASE(114, 10, 4)
        CASE(115, 10, 5)
        CASE(116, 10, 6)
        CASE(117, 10, 7)
        CASE(118, 10, 8)
        CASE(119, 10, 9)
        CASE(120, 10, 10)

        default:
            throw std::runtime_error(
                "Invalid kernel configuration: overlap_scale=" + std::to_string(overlap_scale) +
                ", prefetch_scale=" + std::to_string(prefetch_scale));
    }

#undef CASE
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("rmsnorm", py::overload_cast<torch::Tensor, torch::Tensor, int, int,
//     int>(&rmsnorm_cpp),
//           py::arg("input"), py::arg("weight"), py::arg("overlap_scale"),
//           py::arg("prefetch_scale"), py::arg("overlap_type"));
// }

torch::Tensor gemm(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> out,
                   torch::Tensor D, int overlap_scale, int prefetch_scale) {
    float overlap_ratio = overlap_scale / 10.0;
    float prefetch_ratio = prefetch_scale / 10.0;
    torch::Tensor C;
    if (out.has_value()) {  // Output tensor was provided. So we will use it.
        C = out.value();
    } else {
        const int M = A.sizes()[0];
        const int N = B.sizes()[1];

        auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
        C = torch::empty({M, N}, c_options);
    }
    // Check that all tensors are allocated on GPU device.
    if (!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
        throw std::invalid_argument(
            "gemm only supports GPU device. Use "
            ".to(device=torch.device('cuda'))");
    torch::Tensor _A = A.contiguous();
    torch::Tensor _B = B.contiguous();
    torch::Tensor _C = C.contiguous();
    torch::Tensor _D = D.contiguous();

    gemm_with_prefetch_type_check(_A, _B, _C, _D, overlap_ratio, prefetch_ratio);
    if (!C.is_contiguous()) C.copy_(_C);

    // Return the Torch tensor back to PyTorch
    return C;
}
