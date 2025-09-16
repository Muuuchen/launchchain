#ifndef RESIDUAL_CUH
#define RESIDUAL_CUHa

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "utils.cuh"

// normally residula is done when last kernel is completed

template <typename T, const int OverlapScaled, const int PrefetchScaled, OverlapType overlap_type>
__global__ void add_residual(T* output, const T* input, const T* residual, const int n);

#define INSTANTIATE_RESIDUAL(OVERLAP, PREFETCH, TYPE)                                           \
    template __global__ void add_residual<float, OVERLAP, PREFETCH, TYPE>(float*, const float*, \
                                                                          const float*, int);   \
    template __global__ void add_residual<cutlass::half_t, OVERLAP, PREFETCH, TYPE>(            \
        cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*, int);
#endif