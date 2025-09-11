#include "rmsnorm.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// 计算 RMSNorm 的 GFLOPS
void test_rmsnorm_gflops(int m, int n, float epsilon = 1e-5f, int num_trials = 100) {
    std::cout << "Testing RMSNorm with m=" << m << ", n=" << n << std::endl;
    
    // 计算数组大小
    int n_8 = n / 8;
    int input_size = m * n_8;
    int weight_size = n_8;
    
    // 分配设备内存
    float4 *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float4));
    CUDA_CHECK(cudaGetLastError());
    cudaMalloc(&d_weight, weight_size * sizeof(float4));
    CUDA_CHECK(cudaGetLastError());
    cudaMalloc(&d_output, input_size * sizeof(float4));
    CUDA_CHECK(cudaGetLastError());

    // 估算 FLOPs (浮点运算次数)
    // 第一次循环: 每个元素 2 FLOPs (乘法和加法)
    // 第二次循环: 每个元素 3 FLOPs (乘法、乘法和类型转换)
    // 总共: 5 * m * n FLOPs
    double total_flops = 5.0 * m * n;
    
    // 预热
    rmsnorm_twoPassAlgo_e8<<<m, 256>>>(d_output, d_input, d_weight, m, n, epsilon);
    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 多次运行取平均值
    float total_time = 0.0f;
    for (int i = 0; i < num_trials; i++) {
        cudaEventRecord(start);
        rmsnorm_twoPassAlgo_e8<<<m, 256>>>(d_output, d_input, d_weight, m, n, epsilon);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    
    // 计算平均时间
    float avg_time = total_time / num_trials;
    
    // 计算 GFLOPS
    double gflops = (total_flops / (avg_time * 1e-3)) / 1e9;
    
    // 输出结果
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

// 测试不同形状的 RMSNorm
void test_rmsnorm_performance() {
    std::cout << "RMSNorm Performance Test" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    // 测试不同批次大小
    std::vector<int> batch_sizes = {256};
    
    // 测试不同特征维度 (必须是8的倍数)
    std::vector<int> feature_dims = {4096};
    
    for (int m : batch_sizes) {
        for (int n : feature_dims) {
            test_rmsnorm_gflops(m, n);
        }
    }
    
    // 测试模板化版本 (选择一个代表性的参数组合)
    std::cout << "Testing templated version with OverlapScaled=5, PrefetchScaled=5" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    for (int m : batch_sizes) {
        for (int n : feature_dims) {
            std::cout << "Testing RMSNorm with m=" << m << ", n=" << n << std::endl;
            
            // 分配设备内存
            int n_8 = n / 8;
            int input_size = m * n_8;
            int weight_size = n_8;
            
            float4 *d_input, *d_weight, *d_output;
            cudaMalloc(&d_input, input_size * sizeof(float4));
            cudaMalloc(&d_weight, weight_size * sizeof(float4));
            cudaMalloc(&d_output, input_size * sizeof(float4));
            
            // 估算 FLOPs
            double total_flops = 5.0 * m * n;
            
            // 预热
            rmsnorm_twoPassAlgo_e8_prefetch<5, 5><<<m, 256>>>(d_output, d_input, d_weight, m, n, 1e-5f);
            cudaDeviceSynchronize();
            
            // 创建 CUDA 事件用于计时
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // 多次运行取平均值
            float total_time = 0.0f;
            int num_trials = 100;
            for (int i = 0; i < num_trials; i++) {
                cudaEventRecord(start);
                rmsnorm_twoPassAlgo_e8_prefetch<5, 5><<<m, 256>>>(d_output, d_input, d_weight, m, n, 1e-5f);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                total_time += milliseconds;
            }
            
            // 计算平均时间
            float avg_time = total_time / num_trials;
            
            // 计算 GFLOPS
            double gflops = (total_flops / (avg_time * 1e-3)) / 1e9;
            
            // 输出结果
            std::cout << "Average time: " << avg_time << " ms" << std::endl;
            std::cout << "GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
            std::cout << "-----------------------------" << std::endl;
            
            // 清理
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_input);
            cudaFree(d_weight);
            cudaFree(d_output);
        }
    }
}

int main() {
    test_rmsnorm_gflops(32,1024,1e-5f,1);
    return 0;
}