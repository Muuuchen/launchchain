#include "func.cuh"
#include "rmsnorm.cuh"
#include "utils.cuh"

template <const int OverlapScaled, const int PrefetchScaled,
          OverlapType overlap_type>
__global__ void
rmsnorm_twoPassAlgo_e8_prefetch(float4 *output, const float4 *input,
                                const float4 *weight, const int m, const int n,
                                float epsilon) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  constexpr float overlap_ratio = to_float_device(OverlapScaled);
  constexpr float prefetch_ratio = to_float_device(PrefetchScaled);
  const int overlap_boundary =
      (static_cast<int>(n_8 * prefetch_ratio) / bdimx) * bdimx;
  if constexpr (overlap_type == OverlapType::prefetch) {
    // compute overlap

    // load current block
    uint32_t weight_bytes =
        (uint32_t(n * prefetch_ratio) * sizeof(float) + 0xF) &
        ~0xF; // zeroing last 4 bit
    if (tid == 0) {
      if (weight_bytes % 16 == 0 && weight_bytes <= 0xFFFFFFFF) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                     :
                     : "l"(weight), "r"(weight_bytes)
                     : "memory");
      }
    }
    asm volatile("griddepcontrol.wait;");
  } else if constexpr (overlap_type == OverlapType::shared) {
    ;
  } else {
    ;
  }

  for (int index = tid; index < n_8; index += bdimx) // 0-31   32
  {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;
    local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                     static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                     static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                     static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                     static_cast<float>(h4->y) * static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 weight_val = weight[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&weight_val.x;
    const half2 *g2 = (half2 *)&weight_val.y;
    const half2 *g3 = (half2 *)&weight_val.z;
    const half2 *g4 = (half2 *)&weight_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half2 *h4 = (half2 *)&tmp.w;

    h1->x =
        half(static_cast<float>(l1->x) * s_mean * static_cast<float>(g1->x));
    h1->y =
        half(static_cast<float>(l1->y) * s_mean * static_cast<float>(g1->y));
    h2->x =
        half(static_cast<float>(l2->x) * s_mean * static_cast<float>(g2->x));
    h2->y =
        half(static_cast<float>(l2->y) * s_mean * static_cast<float>(g2->y));
    h3->x =
        half(static_cast<float>(l3->x) * s_mean * static_cast<float>(g3->x));
    h3->y =
        half(static_cast<float>(l3->y) * s_mean * static_cast<float>(g3->y));
    h4->x =
        half(static_cast<float>(l4->x) * s_mean * static_cast<float>(g4->x));
    h4->y =
        half(static_cast<float>(l4->y) * s_mean * static_cast<float>(g4->y));

    output[index] = tmp;
    if constexpr (overlap_type != OverlapType::none)
      if (index >= overlap_boundary)
        asm volatile("griddepcontrol.launch_dependents;");
  }
}

__global__ void rmsnorm_twoPassAlgo_e8(float4 *output, const float4 *input,
                                       const float4 *weight, const int m,
                                       const int n, float epsilon) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;
    local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                     static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                     static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                     static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                     static_cast<float>(h4->y) * static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();
  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 weight_val = weight[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&weight_val.x;
    const half2 *g2 = (half2 *)&weight_val.y;
    const half2 *g3 = (half2 *)&weight_val.z;
    const half2 *g4 = (half2 *)&weight_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half2 *h4 = (half2 *)&tmp.w;

    h1->x =
        half(static_cast<float>(l1->x) * s_mean * static_cast<float>(g1->x));
    h1->y =
        half(static_cast<float>(l1->y) * s_mean * static_cast<float>(g1->y));
    h2->x =
        half(static_cast<float>(l2->x) * s_mean * static_cast<float>(g2->x));
    h2->y =
        half(static_cast<float>(l2->y) * s_mean * static_cast<float>(g2->y));
    h3->x =
        half(static_cast<float>(l3->x) * s_mean * static_cast<float>(g3->x));
    h3->y =
        half(static_cast<float>(l3->y) * s_mean * static_cast<float>(g3->y));
    h4->x =
        half(static_cast<float>(l4->x) * s_mean * static_cast<float>(g4->x));
    h4->y =
        half(static_cast<float>(l4->y) * s_mean * static_cast<float>(g4->y));

    output[index] = tmp;
  }
}

// prefetch
#define INSTANTIATE_TYPE_TRIATS(OVERLAP, TYPE)                                 \
  INSTANTIATE_RMSNORM(OVERLAP, 0, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 1, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 2, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 3, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 4, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 5, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 6, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 7, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 8, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 9, TYPE)                                        \
  INSTANTIATE_RMSNORM(OVERLAP, 10, TYPE)

INSTANTIATE_TYPE_TRIATS(0, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(1, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(2, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(3, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(4, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(5, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(6, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(7, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(8, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(9, OverlapType::prefetch)
INSTANTIATE_TYPE_TRIATS(10, OverlapType::prefetch)

// none
INSTANTIATE_RMSNORM(0, 0, OverlapType::none)

// shared_mem

INSTANTIATE_TYPE_TRIATS(0, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(1, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(2, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(3, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(4, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(5, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(6, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(7, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(8, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(9, OverlapType::shared)
INSTANTIATE_TYPE_TRIATS(10, OverlapType::shared)


