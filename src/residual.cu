#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "residual.cuh"
#include "utils.cuh"
template <typename T, const int OverlapScaled, const int PrefetchScaled, OverlapType overlap_type>
__global__ void add_residual(T* output, const T* input, const T* residual, const int n) {
    extern __shared__ T shared_mem[];
    constexpr float overlap_ratio = to_float_device(OverlapScaled);
    constexpr float prefetch_ratio = to_float_device(PrefetchScaled);
    if constexpr (overlap_type == OverlapType::prefetch) {
        const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
        if (col_index == 0) {
            uint32_t weight_bytes = (uint32_t(n / 2) * sizeof(T)) * prefetch_ratio;
            if (weight_bytes % 16 == 0 && weight_bytes <= 0xFFFFFFFF) {
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                             :
                             : "l"(residual), "r"(weight_bytes)
                             : "memory");
            }
        }
        asm volatile("griddepcontrol.wait;");
        if (col_index < n) {
            output[blockIdx.x * n + col_index] =
                input[blockIdx.x * n + col_index] + residual[blockIdx.x * n + col_index];
        }
        asm volatile("griddepcontrol.launch_dependents;");
    } else if constexpr (overlap_type == OverlapType::shared) {
        const int col_index = blockIdx.y * blockDim.x + threadIdx.x;

        T* weight_s = shared_mem;
        if (col_index < n) {
            weight_s[threadIdx.x] = residual[blockIdx.x * n + col_index];
        }
        asm volatile("griddepcontrol.wait;");
        if (col_index < n) {
            output[blockIdx.x * n + col_index] =
                input[blockIdx.x * n + col_index] + weight_s[threadIdx.x];
        }
        asm volatile("griddepcontrol.launch_dependents;");
    } else {
        const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
        if (col_index < n) {
            output[blockIdx.x * n + col_index] =
                input[blockIdx.x * n + col_index] + residual[blockIdx.x * n + col_index];
        }
    }
}

#define INSTANTIATE_RESIDUAL_TYPE_TRIATS(OVERLAP, TYPE) \
    INSTANTIATE_RESIDUAL(OVERLAP, 0, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 1, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 2, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 3, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 4, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 5, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 6, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 7, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 8, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 9, TYPE)              \
    INSTANTIATE_RESIDUAL(OVERLAP, 10, TYPE)

INSTANTIATE_RESIDUAL_TYPE_TRIATS(0, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(1, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(2, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(3, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(4, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(5, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(6, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(7, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(8, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(9, OverlapType::prefetch)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(10, OverlapType::prefetch)

INSTANTIATE_RESIDUAL(0, 0, OverlapType::none)

INSTANTIATE_RESIDUAL_TYPE_TRIATS(0, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(1, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(2, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(3, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(4, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(5, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(6, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(7, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(8, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(9, OverlapType::shared)
INSTANTIATE_RESIDUAL_TYPE_TRIATS(10, OverlapType::shared)
