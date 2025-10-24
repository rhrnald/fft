
#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>

#include "detail/utils.h"

namespace thunderfft {
// --------------------------------------------
// Precompute twiddle factors (on device memory)
// W_N[k] = exp(-2πi k / N), length = N
// Layout: interleaved real/imag as T[2*N] (Re0, Im0, Re1, Im1, ...)
// Returns device pointer (caller must cudaFree)
// --------------------------------------------
template <typename T> static T *preprocess_W(unsigned int N);

// --------------------------------------------
// Device functions (shared I/O)
//   - Caller allocates __shared__ memory inside a __global__ kernel
//   - s_input / s_output: length = N (vec2_t<T> elements) per signal
//   - W_N: T[2*N] in device memory (global or __constant__)
//   - Batch policy by FFT length N:
//       * N <= 512   : use shared_batch16 (16 signals per block)
//       * N >= 1024  : use shared_batch1  (1 signal  per block)
// --------------------------------------------

template <typename T>
__device__ __forceinline__ void
ThunderFFT_kernel_shared_batch16(vec2_t<T>* __restrict__ s_input,
                                 vec2_t<T>* __restrict__ s_output,
                                 const T*   __restrict__ W_N,
                                 unsigned               N
);

template <typename T>
__device__ __forceinline__ void
ThunderFFT_kernel_shared_batch1(vec2_t<T>* __restrict__ s_input,
                                vec2_t<T>* __restrict__ s_output,
                                const T*   __restrict__ W_N,
                                unsigned               N
);

// --------------------------------------------
// Host wrapper
//   - Internally allocates W_N, launches the FFT, and frees W_N
//   - d_input / d_output length: N * batch (in vec2_t<T> elements)
//   - NO global constraint here (the host may tile/launch as needed)
// --------------------------------------------
template <typename T, unsigned N>
inline void ThunderFFT(vec2_t<T> *d_input, vec2_t<T> *d_output,
                unsigned int batch, cudaStream_t stream = 0);

// --------------------------------------------
// Optional: persistent twiddle management
// --------------------------------------------
template <typename T> void ThunderFFTInitialize(unsigned int N);

template <typename T> void ThunderFFTFinalize();

}  // namespace thunderfft

#include "thunderfft/detail/thunderfft_impl.cuh"