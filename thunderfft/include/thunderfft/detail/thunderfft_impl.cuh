// thunderfft.cu
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "utils.h"

#include "cute/tensor.hpp"

#include "thunderfft/detail/unit_kernel_fp32.cuh"

namespace thunderfft::detail {
// Primary template declaration (must be visible before specialization)
template <typename T, unsigned N, unsigned BATCH>
__device__ __forceinline__
void ThunderFFT_kernel_shared(vec2_t<T>* __restrict__ s_in,
                              vec2_t<T>* __restrict__ s_out,
                              const T*   __restrict__ W_N);

}

// #include "thunderfft/detail/shared_kernel_fp32_n64_b16.cuh"
// #include "thunderfft/detail/shared_kernel_fp32_n4096_b1.cuh"

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

using namespace cute;

template <typename T, unsigned N, unsigned batch_per_block>
static __global__ void ThunderFFT_kernel(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW) {
    extern __shared__ vec2_t<T> smem[];
    constexpr int logN = LOG2P_builtin<N>;
    constexpr int ept = N * batch_per_block / 32;

    auto gmem_layout = make_layout(
        make_shape(
            blockDim.x * batch_per_block,
            Int<N>{}
        ),
        LayoutRight()
    );
    auto input_tensor = make_tensor(make_gmem_ptr(d_input), gmem_layout);
    auto output_tensor = make_tensor(make_gmem_ptr(d_output), gmem_layout);

    auto smem_layout = make_layout(
        make_shape(Int<batch_per_block>{}, Int<N>{}),
        LayoutRight()
    );

    auto smem_tensor = make_tensor(make_smem_ptr(smem), smem_layout);

    auto cta_tiler = make_shape(Int<batch_per_block>{}, Int<N>{});
    auto input_tile = local_tile(input_tensor, cta_tiler, blockIdx.x);
    auto output_tile = local_tile(output_tensor, cta_tiler, blockIdx.x);
    
    // gmem -> smem
    for (int i = 0; i < batch_per_block; i++) {
        smem_tensor(i, threadIdx.x) = input_tile(i, reverse_bit_groups<2,6>(threadIdx.x));
        smem_tensor(i, threadIdx.x + 32) = input_tile(i, reverse_bit_groups<2,6>(threadIdx.x + 32));
    }
    __syncthreads();

    auto reg = make_tensor<vec2_t<T>>(make_shape(Int<ept>{}), LayoutRight());

    // smem -> reg
    for (int i = 0; i < ept / 2; i++) {
        reg(i) = smem_tensor(threadIdx.x/4, i*4 + (threadIdx.x % 4));
        reg(i + ept/2) = smem_tensor(threadIdx.x/4 + batch_per_block/2, i*4 + (threadIdx.x % 4));
    }

    unit::fft_kernel_r64_b16<false>(reinterpret_cast<float*>(reg.data()), dW);
    __syncthreads();

    for (int i = 0; i < ept/2; i++) {
        smem_tensor(threadIdx.x/4, i + (ept/2)*(threadIdx.x % 4)) = reg(i);
        smem_tensor(threadIdx.x/4 + batch_per_block/2, i + (ept/2)*(threadIdx.x % 4)) = reg(i + ept/2);
    }
    __syncthreads();

    // smem -> gmem
    for (int i = 0; i < batch_per_block; i++) {
        output_tile(i, threadIdx.x) = smem_tensor(i, threadIdx.x);
        output_tile(i, threadIdx.x + 32) = smem_tensor(i, threadIdx.x + 32);
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

    
    const size_t shmem_bytes = 2 * sizeof(vec2_t<T>) * (N+pad_h(N)) * batch_per_block;
    cudaFuncSetAttribute(detail::ThunderFFT_kernel<T, N, batch_per_block>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    detail::ThunderFFT_kernel<T, N, batch_per_block><<<grid, block, shmem_bytes, stream>>>(
        d_input, d_output, dW
    );

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    if (!use_cache) cudaFree(dW);
}
} // namespace thunderfft