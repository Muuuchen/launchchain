#ifndef FUNC_CUH
#define FUNC_CUH

// Warp-level reduction
template <typename T, int NUM> __device__ void warpReduceSum(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            val[i] += __shfl_xor_sync(0xffffffff, val[i], mask);
        }
    }
}

// Block-level reductions
template <typename T, int NUM> __device__ void blockReduceSum(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSum<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSum<T, NUM>(val);
}

#endif