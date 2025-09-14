#pragma once
#include <iostream>

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
