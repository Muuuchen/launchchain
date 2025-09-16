#include <ATen/core/TensorBody.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

// 假设你的头文件
#include "binding.hpp"

// CPU版本的RMSNorm作为参考
torch::Tensor rmsnorm_cpu_reference(torch::Tensor input, torch::Tensor weight,
                                    float epsilon = 1e-5) {
    // 转到float32进行CPU计算
    auto input_fp32 = input.to(torch::kFloat32);
    auto weight_fp32 = weight.to(torch::kFloat32);

    // 计算RMS
    auto variance = input_fp32.pow(2).mean(-1, true);
    auto rms = torch::sqrt(variance + epsilon);

    // Normalize and scale
    auto normalized = input_fp32 / rms;
    auto output = normalized * weight_fp32.unsqueeze(0);

    // 转回fp16
    return output.to(torch::kFloat16);
}

// 计算相对误差
float calculate_relative_error(torch::Tensor a, torch::Tensor b) {
    auto diff = (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs();
    auto max_diff = diff.max().item<float>();
    auto mean_diff = diff.mean().item<float>();
    auto b_abs_mean = b.abs().to(torch::kFloat32).mean().item<float>();

    float relative_error = mean_diff / (b_abs_mean + 1e-8);

    std::cout << "  Max absolute error: " << max_diff << std::endl;
    std::cout << "  Mean absolute error: " << mean_diff << std::endl;
    std::cout << "  Relative error: " << relative_error * 100 << "%" << std::endl;

    return relative_error;
}

// 性能测试函数
void benchmark_rmsnorm(int m, int n, int overlap_scale, int prefetch_scale, int overlap_type,
                       int warmup = 10, int iterations = 100) {
    // 创建输入张量
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto input = torch::randn({m, n}, options);
    auto weight = torch::randn({n}, options);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        auto output = rmsnorm_cpp(input, weight, overlap_scale, prefetch_scale, overlap_type);
    }

    // 同步
    torch::cuda::synchronize();

    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto output = rmsnorm_cpp(input, weight, overlap_scale, prefetch_scale, overlap_type);
    }

    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    float avg_time = duration / (float)iterations;

    // 计算带宽 (GB/s)
    // RMSNorm需要读input和weight，写output
    size_t bytes_read = m * n * sizeof(at::Half) + n * sizeof(at::Half);  // input + weight
    size_t bytes_written = m * n * sizeof(at::Half);                      // output
    size_t total_bytes = bytes_read + bytes_written;
    float bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_time / 1e6);

    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
}

// 主测试函数
void test_rmsnorm_correctness() {
    std::cout << "=====================================\n";
    std::cout << "Testing RMSNorm Correctness\n";
    std::cout << "=====================================\n\n";

    // 测试不同的尺寸
    std::vector<std::pair<int, int>> test_sizes = {
        {1, 128},     // 小尺寸
        {32, 512},    // 中等尺寸
        {128, 4096},  // 典型LLM隐藏层尺寸
        {256, 8192},  // 大尺寸
        {1, 12288},   // 单序列，大隐藏维度
    };

    for (auto [m, n] : test_sizes) {
        std::cout << "Testing size: (" << m << ", " << n << ")\n";

        // 创建输入
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto input = torch::randn({m, n}, options);
        auto weight = torch::randn({n}, options);

        // CPU参考实现
        auto cpu_output = rmsnorm_cpu_reference(input, weight);

        // 测试不同的配置
        std::vector<std::tuple<int, int, int>> configs = {
            {0, 0, 0},  // 基础版本
            {5, 5, 1},  // prefetch模式
            {5, 5, 2},  // shared memory模式
        };

        for (auto [overlap_scale, prefetch_scale, overlap_type] : configs) {
            std::cout << "  Config: overlap_scale=" << overlap_scale
                      << ", prefetch_scale=" << prefetch_scale << ", overlap_type=" << overlap_type
                      << std::endl;

            try {
                auto gpu_output =
                    rmsnorm_cpp(input, weight, overlap_scale, prefetch_scale, overlap_type);

                // 检查输出形状
                if (!gpu_output.sizes().equals(input.sizes())) {
                    std::cout << "  ✗ Output shape mismatch!" << std::endl;
                    continue;
                }

                // 计算误差
                float rel_error = calculate_relative_error(gpu_output, cpu_output);

                // 判断是否通过（相对误差小于1%）
                if (rel_error < 0.01) {
                    std::cout << "  ✓ PASSED\n" << std::endl;
                } else {
                    std::cout << "  ✗ FAILED: Error too large\n" << std::endl;
                }

            } catch (const std::exception& e) {
                std::cout << "  ✗ Exception: " << e.what() << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

// 性能测试
void test_rmsnorm_performance() {
    std::cout << "=====================================\n";
    std::cout << "Testing RMSNorm Performance\n";
    std::cout << "=====================================\n\n";

    // 典型的LLM尺寸
    std::vector<std::pair<int, int>> bench_sizes = {
        {128, 4096},  // Llama-7B hidden size
        {128, 8192},  // Llama-70B intermediate size
        {256, 4096},  // Batch size 256
        {512, 4096},  // Batch size 512
    };

    for (auto [m, n] : bench_sizes) {
        std::cout << "Benchmarking size: (" << m << ", " << n << ")\n";

        // 测试不同配置
        std::cout << "  Basic kernel (overlap_type=0):" << std::endl;
        benchmark_rmsnorm(m, n, 0, 0, 0);

        std::cout << "  Prefetch kernel (overlap_type=1):" << std::endl;
        benchmark_rmsnorm(m, n, 5, 5, 1);

        std::cout << "  Shared memory kernel (overlap_type=2):" << std::endl;
        benchmark_rmsnorm(m, n, 5, 5, 2);

        std::cout << std::endl;
    }
}

// 边界条件测试
void test_edge_cases() {
    std::cout << "=====================================\n";
    std::cout << "Testing Edge Cases\n";
    std::cout << "=====================================\n\n";

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

    // 测试极小尺寸
    std::cout << "Testing very small size (1, 8):" << std::endl;
    try {
        auto input = torch::randn({1, 8}, options);
        auto weight = torch::randn({8}, options);
        auto output = rmsnorm_cpp(input, weight, 0, 0, 0);
        std::cout << "  ✓ Small size works\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  ✗ Failed: " << e.what() << std::endl;
    }

    // 测试非8的倍数
    std::cout << "Testing non-aligned size (3, 100):" << std::endl;
    try {
        auto input = torch::randn({3, 100}, options);
        auto weight = torch::randn({100}, options);
        auto output = rmsnorm_cpp(input, weight, 0, 0, 0);
        std::cout << "  ✓ Non-aligned size works\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  ✗ Failed: " << e.what() << std::endl;
    }

    // 测试特殊值
    std::cout << "Testing special values (zeros, ones, inf):" << std::endl;
    try {
        // 全零输入
        auto zero_input = torch::zeros({32, 128}, options);
        auto weight = torch::ones({128}, options);
        auto zero_output = rmsnorm_cpp(zero_input, weight, 0, 0, 0);

        // 检查输出是否为零
        if (zero_output.abs().max().item<float>() < 1e-5) {
            std::cout << "  ✓ Zero input handled correctly" << std::endl;
        } else {
            std::cout << "  ✗ Zero input produces non-zero output" << std::endl;
        }

        // 包含inf的输入
        auto inf_input = torch::randn({32, 128}, options);
        inf_input[0][0] = std::numeric_limits<float>::infinity();
        auto inf_output = rmsnorm_cpp(inf_input, weight, 0, 0, 0);

        // 检查是否有NaN
        if (!torch::any(torch::isnan(inf_output)).item<bool>()) {
            std::cout << "  ✓ Inf input handled without NaN" << std::endl;
        } else {
            std::cout << "  ✗ Inf input produces NaN" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  ✗ Special values test failed: " << e.what() << std::endl;
    }
}

// 参数范围测试
void test_parameter_ranges() {
    std::cout << "=====================================\n";
    std::cout << "Testing Parameter Ranges\n";
    std::cout << "=====================================\n\n";

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto input = torch::randn({32, 512}, options);
    auto weight = torch::randn({512}, options);

    // 测试overlap_scale范围 (0-10)
    std::cout << "Testing overlap_scale range:" << std::endl;
    for (int os = 0; os <= 10; os += 2) {
        try {
            auto output = rmsnorm_cpp(input, weight, os, 5, 1);
            std::cout << "  overlap_scale=" << os << " ✓" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  overlap_scale=" << os << " ✗ " << e.what() << std::endl;
        }
    }

    std::cout << "\nTesting prefetch_scale range:" << std::endl;
    // 测试prefetch_scale范围 (0-10)
    for (int ps = 0; ps <= 10; ps += 2) {
        try {
            auto output = rmsnorm_cpp(input, weight, 5, ps, 1);
            std::cout << "  prefetch_scale=" << ps << " ✓" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  prefetch_scale=" << ps << " ✗ " << e.what() << std::endl;
        }
    }
}

int main() {
    // 检查CUDA是否可用
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }

    std::cout << "CUDA Device: " << torch::cuda::device_count() << " available" << std::endl;
    std::cout << "Current Device: " << torch::cuda::device_count() << std::endl;

    // 设置随机种子以保证可重复性
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);

    try {
        // 运行所有测试
        test_rmsnorm_correctness();
        test_edge_cases();
        test_parameter_ranges();
        test_rmsnorm_performance();

        std::cout << "\n=====================================\n";
        std::cout << "All tests completed!\n";
        std::cout << "=====================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}