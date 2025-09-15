#pragma once
#include <iostream>
#include "utils.h"
#include "stat.h"

#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }

template <typename Kernel>
float measure_execution_ms(Kernel &&kernel, const unsigned int warm_up_runs,
                           const unsigned int runs, cudaStream_t stream) {
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    for (size_t i = 0; i < warm_up_runs; i++) {
        kernel(stream);
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
    for (size_t i = 0; i < runs; i++) {
        kernel(stream);
    }
    CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    float time;
    CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
    return time;
}

template <typename T>
__host__ __device__ constexpr const char *bench_type_cstr() {
    if constexpr (std::is_same_v<T, half>)
        return "half";
    if constexpr (std::is_same_v<T, float>)
        return "float";
    if constexpr (std::is_same_v<T, double>)
        return "double";
    else
        return "unknown";
}

template <typename T, unsigned int N, unsigned int radix, typename Validator, typename Benchmark>
void benchmark_run(
    Validator&& validator, Benchmark&& benchmark, vec2_t<T> *h_data, float2 *baseline, unsigned int B
) {
    using T2 = vec2_t<T>;

    T2 *d_custom = nullptr;
    CHECK_CUDA(cudaMalloc(&d_custom, sizeof(T2) * N * B));
    CHECK_CUDA(cudaMemcpy(d_custom, h_data, sizeof(T2) * N * B,
                          cudaMemcpyHostToDevice));

    validator(d_custom, B);

    T2 *h_custom = static_cast<T2 *>(std::malloc(sizeof(T2) * N));
    CHECK_CUDA(cudaMemcpy(h_custom, d_custom, sizeof(T2) * N,
                          cudaMemcpyDeviceToHost));

    const double max_err =
        static_cast<double>(check_max_abs_err(baseline, h_custom, N));

    const PerfStat perf = benchmark(d_custom, B);

    stat::push(stat::RunStat{bench_type_cstr<T>(), N, radix, B,
                             max_err, perf.comp_ms, perf.comm_ms, perf.e2e_ms});

    CHECK_CUDA(cudaFree(d_custom));
    std::free(h_custom);
}