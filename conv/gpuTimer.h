#pragma once
#include <cuda_runtime.h>

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t stream;

    GpuTimer(cudaStream_t stream) : stream(stream) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start, stream); }

    void Stop() { cudaEventRecord(stop, stream); }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
