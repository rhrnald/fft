// sample/main.cu (float & float2 only)
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <thunderfft/thunderfft.cuh>  // ThunderFFT<T>, ThunderFFTInitialize/Finalize

#ifndef CHECK_CUDA
#define CHECK_CUDA(stmt)                                                       \
    do {                                                                       \
        cudaError_t err__ = (stmt);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// Generate a simple test signal on host: h has length N*batch
static void make_test_input(float2* h, unsigned N, unsigned batch) {
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    for (unsigned b = 0; b < batch; ++b) {
        const float f0  = static_cast<float>(1 + b);
        const float amp = 0.5f + 0.1f * static_cast<float>(b);
        for (unsigned i = 0; i < N; ++i) {
            const float t  = static_cast<float>(i) / static_cast<float>(N);
            const float re = amp * std::cos(two_pi * f0 * t)
                           + 0.1f * std::cos(two_pi * 7.0f * t);
            const float im = amp * std::sin(two_pi * f0 * t)
                           + 0.1f * std::sin(two_pi * 7.0f * t);
            h[b * N + i] = make_float2(re, im);
        }
    }
}

// Simple checksum to sanity-check output content
static double checksum(const float2* v, unsigned N, unsigned batch) {
    long double acc = 0.0L;
    const size_t total = static_cast<size_t>(N) * batch;
    for (size_t i = 0; i < total; ++i) {
        acc += static_cast<long double>(v[i].x) * 1.0001L
             + static_cast<long double>(v[i].y) * 0.9997L;
    }
    return static_cast<double>(acc);
}

int main(int argc, char** argv) {
    constexpr unsigned N     = 64;        // default size
    unsigned batch = 65536;     // default batch
    int device     = 0;

    // Parse CLI: main [N] [batch] [device]
    // if (argc >= 2) N     = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    // if (argc >= 3) batch = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    // if (argc >= 4) device= std::atoi(argv[3]);

    CHECK_CUDA(cudaSetDevice(device));
    std::cout << "ThunderFFT sample (float/float2)\n"
              << "  Type   : float\n"
              << "  N      : " << N << "\n"
              << "  batch  : " << batch << "\n"
              << "  device : " << device << "\n";

    // Host buffers (use pinned memory for faster H2D/D2H if desired)
    const size_t count = static_cast<size_t>(N) * batch;
    const size_t bytes = sizeof(float2) * count;

    float2* h_in  = static_cast<float2*>(std::malloc(bytes));
    float2* h_out = static_cast<float2*>(std::malloc(bytes));
    if (!h_in || !h_out) {
        std::fprintf(stderr, "Host allocation failed (%zu bytes)\n", bytes);
        return EXIT_FAILURE;
    }

    make_test_input(h_in, N, batch);

    // Device buffers
    float2* d_in  = nullptr;
    float2* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Optional: build and cache twiddles once (recommended for repeated runs)
    thunderfft::ThunderFFTInitialize<float>(N);

    // Warm-up
    thunderfft::ThunderFFT<float, N>(d_in, d_out, batch, /*stream=*/0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 5;
    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        thunderfft::ThunderFFT<float, N>(d_in, d_out, batch, /*stream=*/0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= static_cast<float>(iters); // average per call

    // Copy back and print a few values
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average time (ThunderFFT): " << ms << " ms\n";
    std::cout << "Checksum(out): " << checksum(h_out, N, batch) << "\n";

    const unsigned to_print = std::min<unsigned>(8, N);
    std::cout << "First " << to_print << " outputs of batch 0:\n";
    for (unsigned i = 0; i < to_print; ++i) {
        const float2 z = h_out[i];
        std::cout << "  y[" << i << "] = (" << z.x << ", " << z.y << ")\n";
    }

    // Cleanup
    thunderfft::ThunderFFTFinalize<float>();
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    std::free(h_in);
    std::free(h_out);

    std::cout << "Done.\n";
    return 0;
}
