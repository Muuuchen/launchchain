#include "utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>


__global__ void rmsnorm_twoPassAlgo_e8(float4* output, const float4* input, const float4* weight,
    const int m, const int n, float epsilon);

//  If the value is not a multiple of 16, then the behavior is undefined. The address srcMem must be
//  aligned to 16 bytes.
template <const int OverlapScaled, const int PrefetchScaled>
__global__ void rmsnorm_twoPassAlgo_e8_prefetch(float4* output, const float4* input,
                                                const float4* weight, const int m, const int n,
                                                float epsilon);



#define INSTANTIATE_RMSNORM(OVERLAP, PREFETCH) \
    template __global__ void rmsnorm_twoPassAlgo_e8_prefetch<OVERLAP, PREFETCH> \
    (float4*, const float4*, const float4*, int, int, float); \
    static_assert(true, "Instantiated rmsnorm_twoPassAlgo_e8_prefetch<" #OVERLAP ", " #PREFETCH ">"); 



