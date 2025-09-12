#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "helper.h"
#include "stat.h"

#include "fft_tc_sm.h"

namespace {

// 내부용 타입 문자열 (half/float/double)
template <typename T>
__host__ __device__ constexpr const char* bench_type_cstr() {
    if constexpr (std::is_same_v<T, half>)   return "half";
    if constexpr (std::is_same_v<T, float>)  return "float";
    if constexpr (std::is_same_v<T, double>) return "double";
    else return "unknown";
}

struct PerfStat { double comp_ms; double e2e_ms; };

} // namespace

template<typename T, unsigned int N, unsigned int radix>
static inline PerfStat fft_tc_sm_perf(vec2_t<T>* d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1000;
    static constexpr unsigned int kernel_runs    = 10;
    static constexpr unsigned int warm_up_runs   = 1;
    static constexpr unsigned int warp_per_block = 1;

    int bank_padding = 9;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // R 회와 2R 회를 이용해 per-iteration kernel-only 시간 산출
    double t_R = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block),
                   dim3(32, warp_per_block),
                   16 * (N + bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_2R = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block),
                   dim3(32, warp_per_block),
                   16 * (N + bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, 2 * inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const double comp_ms =
        (t_2R - t_R) / static_cast<double>(inside_repeats);

    // end-to-end(런치 오버헤드 포함, 반복 1회)
    double e2e_ms = measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block),
                   dim3(32, warp_per_block),
                   16 * (N + bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, 1);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaStreamDestroy(stream));
    return {comp_ms, e2e_ms};
}

template<typename T, unsigned int N, unsigned int radix>
static inline void fft_tc_sm_val(vec2_t<T>* d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs    = 1;
    static constexpr unsigned int warm_up_runs   = 0;
    static constexpr unsigned int warp_per_block = 1;

    int bank_padding = 9;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    (void)measure_execution_ms(
        [&](cudaStream_t s) {
            kernel_fft_tc_sm<T, N, radix, false>
                <<<B / (16 * warp_per_block),
                   dim3(32, warp_per_block),
                   16 * (N + bank_padding) * sizeof(T) * 2 * warp_per_block,
                   s>>>(d_data, inside_repeats);
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaStreamDestroy(stream));
}

template<typename T, unsigned int N, unsigned int radix>
void fft_tc_sm_run(vec2_t<T>* h_data, float2* baseline, unsigned int B) {
    using T2 = vec2_t<T>;

    // H2D
    T2* d_custom = nullptr;
    CHECK_CUDA(cudaMalloc(&d_custom, sizeof(T2) * N * B));
    CHECK_CUDA(cudaMemcpy(d_custom, h_data, sizeof(T2) * N * B, cudaMemcpyHostToDevice));

    // 검증용 실행
    fft_tc_sm_val<T, N, radix>(d_custom, B);

    // 첫 배치 N개만 D2H해서 오차 계산
    T2* h_custom = static_cast<T2*>(std::malloc(sizeof(T2) * N));
    CHECK_CUDA(cudaMemcpy(h_custom, d_custom, sizeof(T2) * N, cudaMemcpyDeviceToHost));

    const double max_err = static_cast<double>(check_max_abs_err(baseline, h_custom, N));

    // 성능 측정
    const auto perf = fft_tc_sm_perf<T, N, radix>(d_custom, B);

    // 결과 누적 (출력은 메인에서)
    stat::push(stat::RunStat{
        /*type*/    bench_type_cstr<T>(),
        /*N*/       N,
        /*radix*/   radix,
        /*B*/       B,
        /*max_err*/ max_err,
        /*comp_ms*/ perf.comp_ms,
        /*e2e_ms*/  perf.e2e_ms
    });

    // 정리
    CHECK_CUDA(cudaFree(d_custom));
    std::free(h_custom);
}

template<unsigned int N>
void fft_tc_sm_benchmark(float2 *h_input, half2 * h_input_half, float2 *baseline, int batch) {
    fft_tc_sm_run<half, N, 8>(h_input_half, baseline, batch);
    // fft_tc_sm_run<half, N, 16>(h_input_half, baseline, batch);
    fft_tc_sm_run<float, N, 8>(h_input, baseline, batch);
    // fft_tc_sm_run<float, N, 16>(h_input, baseline, batch);
}