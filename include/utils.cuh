#ifndef UTILS_CUH
#define UTILS_CUH
#include <cstdio>

//    CUDA_CHECK(cudaGetLastError());
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__);            \
      fprintf(stderr, "    Call: %s\n", #call);                                \
      fprintf(stderr, "    Error: %s (%d)\n", cudaGetErrorString(error_code),  \
              error_code);                                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int SCALE = 10; // 0.0 0.1 → 1, 0.2 → 2, ..., 1.0 → 10
constexpr int MIN_VAL = 0;
constexpr int MAX_VAL = 10;
// 0.0 means none ratio
//               overlap ratio                    prefetch_ratio
// 0.0        none overlap(disable pdl)            prefetch_noting
// 1.0      in loop how early to overlap           prefetch all
// lager overlap ratio means early in loop

constexpr float to_float(int scaled) {
  return static_cast<float>(scaled) / SCALE;
}
__device__ constexpr float to_float_device(int scaled) {
  return static_cast<float>(scaled) / SCALE;
}

enum class OverlapType { none, prefetch, shared };
#endif