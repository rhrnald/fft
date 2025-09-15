#pragma once

#include <cassert>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "helper.h"
#include "stat.h"

#include "fft_tc_sm.h"

template <typename T, unsigned int N, unsigned int radix>
static inline PerfStat fft_tc_sm_perf(vec2_t<T> *d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1000;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 1;
    static constexpr unsigned int warp_per_block = 1;

    int max_bank_padding = 16;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // R 회와 2R 회를 이용해 per-iteration kernel-only 시간 산출
    double t_R = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
                   16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_2R = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
                   16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, 2 * inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const double comp_ms = (t_2R - t_R) / static_cast<double>(inside_repeats);

    // end-to-end(런치 오버헤드 포함, 반복 1회)
    double e2e_ms = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
                   16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, 1);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double comm_ms = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
                    16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
                    s>>>(d_data, 0);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaStreamDestroy(stream));
    return {comp_ms, e2e_ms, comm_ms};
}

template <typename T, unsigned int N, unsigned int radix>
static inline void fft_tc_sm_val(vec2_t<T> *d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 0;
    static constexpr unsigned int warp_per_block = 1;

    int max_bank_padding = 16;



    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    (void)measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
                   16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaStreamDestroy(stream));
}

template <unsigned int N>
void fft_tc_sm_benchmark(float2 *h_input, half2 *h_input_half, float2 *baseline,
                         int batch) {
    benchmark_run<half, N, 8>(
        [&](half2* d_custom, int B) {
            fft_tc_sm_val<half, N, 8>(d_custom, B);
        }, 
        [&](half2* d_custom, int B) {
            return fft_tc_sm_perf<half, N, 8>(d_custom, B);
        },
        h_input_half, baseline, batch);

    benchmark_run<float, N, 8>(
        [&](float2* d_custom, int B) {
            fft_tc_sm_val<float, N, 8>(d_custom, B);
        }, 
        [&](float2* d_custom, int B) {
            return fft_tc_sm_perf<float, N, 8>(d_custom, B);
        },
        h_input, baseline, batch);
    // fft_tc_sm_run<half, N, 8>(h_input_half, baseline, batch);
    // fft_tc_sm_run<half, N, 16>(h_input_half, baseline, batch);
    // fft_tc_sm_run<float, N, 8>(h_input, baseline, batch);

    // fft_tc_sm_run<float, N, 16>(h_input, baseline, batch);

    free(h_input);
    free(h_input_half);
}