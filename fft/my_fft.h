#pragma once

#include <cassert>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "utils.h"
#include "helper.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

__global__ void
fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64,
                           unsigned int repeat);
__global__ void fft_kernel_radix64_batch16_half(half2 *d_data,
                                                const half2 *__restrict__ W_64,
                                                unsigned int repeat);
__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <typename T, unsigned int N>
static inline PerfStat my_fft_perf(vec2_t<T> *d_data, int batch) {
    static constexpr unsigned int inside_repeats = 10000;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 1;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    using T2 = std::conditional_t<std::is_same_v<T, float>, float2, half2>;
    T2 h_W_64[64];
    // T h_W_4096[4096];

    auto make_val = [](float re, float im) {
        if constexpr (std::is_same_v<T, float>) {
            return make_float2(re, im);
        } else if constexpr (std::is_same_v<T, half>) {
            // float -> half2 변환 (round-to-nearest-even)
            return __floats2half2_rn(re, im);
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported T");
        }
    };

    for (int i = 0; i < 64; ++i) {
        float theta = -2.0f * PI * i / 64.0f;
        h_W_64[i] = make_val(cosf(theta), sinf(theta));
    }

    // for (int i = 0; i < 4096; ++i) {
    //     float theta = -2.0f * PI * i / 4096.0f;
    //     h_W_4096[i] = make_val(cosf(theta), sinf(theta));
    // }

    vec2_t<T> *W_64, *W_4096;
    CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(T2)));
    // CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(T)));
    CHECK_CUDA(
        cudaMemcpy(W_64, h_W_64, 64 * sizeof(T2), cudaMemcpyHostToDevice));
    // CHECK_CUDA(
    //     cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFuncSetAttribute(fft_kernel_radix64_batch16, cudaFuncAttributePreferredSharedMemoryCarveout, 75);

    // R 회와 2R 회를 이용해 per-iteration kernel-only 시간 산출
    double t_R = measure_execution_ms(
        [&](cudaStream_t stream) {
            if constexpr (std::is_same_v<T, float>) {
                fft_kernel_radix64_batch16<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                    d_data, W_64, inside_repeats);
            } else if constexpr (std::is_same_v<T, half>) {
                fft_kernel_radix64_batch16_half<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                    d_data, W_64, inside_repeats);
                assert("4096 half is not supported" && false);
            }
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_2R = measure_execution_ms(
        [&](cudaStream_t stream) {
            if constexpr (std::is_same_v<T, float>) {
                fft_kernel_radix64_batch16<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                    d_data, W_64, 2 * inside_repeats);
            } else if constexpr (std::is_same_v<T, half>) {
                fft_kernel_radix64_batch16_half<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                    d_data, W_64, 2 * inside_repeats);
                assert("4096 half is not supported" && false);
            }
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    double comp_ms = (t_2R - t_R) / static_cast<double>(inside_repeats);

    double e2e_ms = measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_kernel_radix64_batch16<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                d_data, W_64, 1);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    double comm_ms = measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_kernel_radix64_batch16<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), stream>>>(
                d_data, W_64, 0);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaStreamDestroy(stream));
    return {comp_ms, e2e_ms, comm_ms};
}

template <typename T, unsigned int N>
static inline void my_fft_val(vec2_t<T> *d_data, int batch) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 0;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaFuncSetAttribute(fft_kernel_radix64_batch16, cudaFuncAttributePreferredSharedMemoryCarveout, 75);

    using T2 = std::conditional_t<std::is_same_v<T, float>, float2, half2>;
    T2 h_W_64[64];
    // T h_W_4096[4096];

    auto make_val = [](float re, float im) {
        if constexpr (std::is_same_v<T, float>) {
            return make_float2(re, im);
        } else if constexpr (std::is_same_v<T, half>) {
            // float -> half2 변환 (round-to-nearest-even)
            return __floats2half2_rn(re, im);
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported T");
        }
    };

    for (int i = 0; i < 64; ++i) {
        float theta = -2.0f * PI * i / 64.0f;
        h_W_64[i] = make_val(cosf(theta), sinf(theta));
    }

    T2 *W_64;
    CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(T2)));
    CHECK_CUDA(
        cudaMemcpy(W_64, h_W_64, 64 * sizeof(T2), cudaMemcpyHostToDevice));

    (void)measure_execution_ms(
        [&](cudaStream_t s) {
            fft_kernel_radix64_batch16<<<batch / 16, 32, 32 * 2 * (32/2 + 1) * sizeof(T2), s>>>(
                d_data, W_64, inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaStreamDestroy(stream));
}

template <unsigned int N>
void my_fft_benchmark(float2 *h_input, half2 *h_input_half, float2 *baseline,
                         int batch) {
    benchmark_run<float, N, 4>(
        [&](float2* d_custom, int B) {
            my_fft_val<float, N>(d_custom, B);
        }, 
        [&](float2* d_custom, int B) {
            return my_fft_perf<float, N>(d_custom, B);
        },
        h_input, baseline, batch);
}
