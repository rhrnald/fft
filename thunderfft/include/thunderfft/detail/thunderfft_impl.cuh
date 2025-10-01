// thunderfft.cu
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "utils.h"


#include "thunderfft/detail/shared_kernel_fp32_n64_b16.cuh"

namespace thunderfft::detail {

template <typename T, unsigned N, unsigned BATCH>
__device__ __forceinline__
void ThunderFFT_kernel_shared(vec2_t<T>* __restrict__ s_in,
                              vec2_t<T>* __restrict__ s_out,
                              const T*   __restrict__ W_N);

// ===============================
// Shared device kernels 
// ===============================

// ===================================
// Global kernels (implemented)
// ===================================

template <typename T, unsigned N, unsigned batch>
static __global__ void ThunderFFT_kernel(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW) {
    extern __shared__ unsigned char smem_raw[];
    vec2_t<T>* s_in  = reinterpret_cast<vec2_t<T>*>(smem_raw);
    vec2_t<T>* s_out = s_in + N;

    const unsigned b = blockIdx.x;

    // gmem -> smem
    for (unsigned i = threadIdx.x; i < N * batch; i += blockDim.x) {
        s_in[i] = d_input[static_cast<size_t>(b) * N * batch + i];
    }
    __syncthreads();

    // Shared compute
    ThunderFFT_kernel_shared<T, N, batch>(s_in, s_out, dW);

    // smem -> gmem
    for (unsigned i = threadIdx.x; i < N * batch; i += blockDim.x) {
        d_output[static_cast<size_t>(b) * N * batch + i] = s_out[i];
    }
}

} // namespace thunderfft::detail


namespace thunderfft {

template <typename T>
struct TwiddleCache {
    static T* dW;
    static unsigned N_cached;
};
template <typename T> T*       TwiddleCache<T>::dW       = nullptr;
template <typename T> unsigned TwiddleCache<T>::N_cached = 0;

// ==================================
// Twiddle builders (with half spec.)
// ==================================
// Store only cos table: W_N[k] = cos(2π k / N), length = N
template <typename T>
inline T* preprocess_W(unsigned int N) {
    if (N == 0) return nullptr;

    std::vector<T> h(N);
    const double two_pi_over_N = 2.0 * M_PI / static_cast<double>(N);

    for (unsigned k = 0; k < N; ++k) {
        const double ang = two_pi_over_N * static_cast<double>(k);
        h[k] = static_cast<T>(std::cos(ang));
    }

    T* d = nullptr;
    tf_check_cuda(cudaMalloc(&d, sizeof(T) * N), "cudaMalloc twiddles (cos only)");
    tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(T) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy twiddles (cos only) H2D");
    return d;
}

// Specialization for half (__half). Assumes 'half' typedef/using is available.
template <>
inline half* preprocess_W<half>(unsigned int N) {
    if (N == 0) return nullptr;

    std::vector<half> h(N);
    const double two_pi_over_N = 2.0 * M_PI / static_cast<double>(N);

    for (unsigned k = 0; k < N; ++k) {
        const double ang = two_pi_over_N * static_cast<double>(k);
        const float  c  = static_cast<float>(std::cos(ang));
        h[k] = __float2half(c);
    }

    half* d = nullptr;
    tf_check_cuda(cudaMalloc(&d, sizeof(half) * N), "cudaMalloc twiddles (cos only, half)");
    tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(half) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy twiddles (cos only, half) H2D");
    return d;
}


// =========================
// Twiddle cache management
// =========================
template <typename T>
inline void ThunderFFTInitialize(unsigned int N) {
    if (TwiddleCache<T>::dW && TwiddleCache<T>::N_cached == N) return;
    if (TwiddleCache<T>::dW) {
        cudaFree(TwiddleCache<T>::dW);
        TwiddleCache<T>::dW = nullptr;
        TwiddleCache<T>::N_cached = 0;
    }
    TwiddleCache<T>::dW = preprocess_W<T>(N);
    TwiddleCache<T>::N_cached = N;
}

template <typename T>
inline void ThunderFFTFinalize() {
    if (TwiddleCache<T>::dW) {
        cudaFree(TwiddleCache<T>::dW);
        TwiddleCache<T>::dW = nullptr;
        TwiddleCache<T>::N_cached = 0;
    }
}

// ============================
// Host wrapper (grid policy)
// ============================
template <typename T, unsigned N>
inline void ThunderFFT(vec2_t<T>* d_input,
                vec2_t<T>* d_output,
                unsigned   batch,
                cudaStream_t stream) {
    if (N == 0 || batch == 0) return;

    // Resolve twiddle source (use cache if available)
    T* dW = nullptr;
    const bool use_cache = (TwiddleCache<T>::dW && TwiddleCache<T>::N_cached == N);
    dW = use_cache ? TwiddleCache<T>::dW : preprocess_W<T>(N);

    // Shared memory: [in | out], each N elements

    constexpr unsigned batch_per_block = (N <= 512u ? 16u : 1u);
    constexpr unsigned threads_per_warp = 32;

    const dim3 grid(batch / batch_per_block );
    const dim3 block(threads_per_warp);

    
    const size_t shmem_bytes = sizeof(vec2_t<T>) * N * batch_per_block;

    detail::ThunderFFT_kernel<T, N, batch_per_block><<<grid, block, shmem_bytes, stream>>>(
        d_input, d_output, dW
    );

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    if (!use_cache) cudaFree(dW);
}
} // namespace thunderfft