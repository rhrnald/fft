
#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>

#include "detail/util_cuda.h"

namespace thunderfft {

// Logical indexing:
//   i : element index within a signal
//   j : batch index
//
// Linear logical index (no padding):
//   idx = i * elem_stride + j * batch_stride
//
// Physical index with periodic padding:
//   - Every pad_period logical elements form one block
//   - Each block is followed by 'pad' padding elements
//
//   padded_idx = idx + (idx / pad_period) * pad

template<int N_, int Batch_, int ElemStride_, int BatchStride_, int PadPeriod_, int Pad_, bool Reversed_>
struct layout_t {
  static constexpr int N           = N_;
  static constexpr int batch       = Batch_;
  static constexpr int elem_stride = ElemStride_;
  static constexpr int batch_stride= BatchStride_;
  static constexpr int pad_period  = PadPeriod_;
  static constexpr int pad         = Pad_;
  static constexpr int reversed     = Reversed_;
};


// ============================================================================
// Twiddle factor management
// ============================================================================
//
// Twiddle definition:
//   W_N[k] = exp(-2πi k / N), k = 0..N-1
//
// Storage format (device memory):
//   - Interleaved real/imag: T[2*N]
//   - Layout: (Re0, Im0, Re1, Im1, ...)
//
// Ownership:
//   - preprocess_W returns a device pointer (caller must cudaFree).
//   - ThunderFFTInitialize/Finalize optionally manage persistent twiddle storage.
// ============================================================================

// Optional: persistent twiddle management.
// - Initialize internal device twiddle tables for the given N.
// - Intended to be called once per N (or per application lifecycle).
template <typename T>
void ThunderFFTInitialize(int N);

// Optional: release internal twiddle tables (if managed persistently).
template <typename T>
void ThunderFFTFinalize();

// Allocate and precompute W_N on device memory.
// - Returns: device pointer to T[2*N] (caller must cudaFree).
template <typename T>
static T* preprocess_W(int N);


// ============================================================================
// Device utilities: data movement (gmem/smem <-> registers)
// ============================================================================
//
// Conventions:
//   - Argument order: (dst, src) to match memcpy-style semantics.
//   - __restrict__ indicates no-aliasing assumptions for better optimization.
//   - These helpers are intended to be used inside __global__ kernels.
//
// Notes on layout:
//   - Shared-memory helpers (smem2reg/reg2smem and gmem<->smem variants) use
//     the compile-time layout parameter (sL) to map indices and handle padding.
// ============================================================================


// --------------------------------------------
// Global memory -> register file
// --------------------------------------------
// Loads one logical FFT fragment from global memory into per-thread registers.
// - reg  : destination register buffer
// - gmem : source global memory buffer (read-only)
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_gmem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ gmem);


// --------------------------------------------
// Register file -> global memory
// --------------------------------------------
// Stores one logical FFT fragment from per-thread registers to global memory.
// - gmem : destination global memory buffer
// - reg  : source register buffer (read-only)
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_reg2gmem(vec2_t<T>* __restrict__ gmem,
                    const vec2_t<T>* __restrict__ reg);


// --------------------------------------------
// Shared memory -> register file (layout-aware)
// --------------------------------------------
// Loads from shared memory into per-thread registers using layout sL.
// Typical use: staged I/O between FFT stages or transpose/reorder steps.
// - reg  : destination register buffer
// - smem : source shared memory buffer (read-only)
// - sL   : compile-time layout describing (N, batch, block, pad) and indexing
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);


// --------------------------------------------
// Register file -> shared memory (layout-aware)
// --------------------------------------------
// Stores per-thread registers into shared memory using layout sL.
// - smem : destination shared memory buffer
// - reg  : source register buffer (read-only)
// - sL   : compile-time layout describing (N, batch, block, pad) and indexing
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);


// --------------------------------------------
// Global memory -> shared memory (layout-aware)
// --------------------------------------------
// Directly stages data from global memory into shared memory using layout sL.
// Intended for kernels that operate primarily out of shared memory.
// - smem : destination shared memory buffer
// - gmem : source global memory buffer (read-only)
// - sL   : compile-time layout for shared placement (including padding/stride)
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_gmem2smem(vec2_t<T>* __restrict__ smem,
                     const vec2_t<T>* __restrict__ gmem);


// --------------------------------------------
// Shared memory -> global memory (layout-aware)
// --------------------------------------------
// Writes staged results in shared memory back to global memory.
// - gmem : destination global memory buffer
// - smem : source shared memory buffer (read-only)
// - sL   : compile-time layout for shared placement (including padding/stride)
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2gmem(vec2_t<T>* __restrict__ gmem,
                     const vec2_t<T>* __restrict__ smem);


// ============================================================================
// Public API: launch wrapper (global I/O)
// ============================================================================
//
// Batch policy (by FFT length N):
//   - N <= 512  : shared_batch16 (16 signals per block)
//   - N >= 1024 : shared_batch1  (1 signal per block)
//
// d_input / d_output:
//   - Arrays of vec2_t<T> in global memory
//   - Each signal has length N (vec2_t<T> elements)
// ============================================================================

template <typename T, int N, bool forward = true>
inline void ThunderFFT_global(vec2_t<T>* d_input,
                              vec2_t<T>* d_output,
                              int batch,
                              cudaStream_t stream = 0);


// ============================================================================
// FFT kernels: shared-memory / register variants
// ============================================================================

// --------------------------------------------
// Shared-memory kernel (no padding layout assumed)
// --------------------------------------------
// Performs FFT using shared-memory I/O buffers.
// - s_input / s_output: shared-memory buffers, length N per signal
// - W_N: twiddle table for length N (device pointer to T[2*N])
// - W_64: optional specialized twiddle table for radix-64 (device pointer)
// - N, batch: runtime parameters used for dispatch/loops as needed
template <typename T>
__device__ __forceinline__ void
ThunderFFT_kernel_smem(vec2_t<T>* __restrict__ s_input,
                       vec2_t<T>* __restrict__ s_output,
                       const T* __restrict__ W_N,
                       int N,
                       int batch);


// --------------------------------------------
// Shared-memory kernel (layout-aware padding)
// --------------------------------------------
// Same as ThunderFFT_kernel_smem but the shared-memory I/O uses explicit
// compile-time layouts for input and output to handle padded/strided layouts.
// - input_L / output_L: compile-time layouts describing shared placement
template <typename T, typename input_L, typename output_L>
__device__ __forceinline__ void
ThunderFFT_kernel_smem_pad(vec2_t<T>* __restrict__ s_input,
                           vec2_t<T>* __restrict__ s_output,
                           const T* __restrict__ W_N,
                           int N,
                           int batch);


// --------------------------------------------
// Register-resident kernel
// --------------------------------------------
// Performs FFT primarily in registers (caller provides reg buffer).
// - reg: per-thread register buffer containing the working set
// - W_N / W_64: twiddle tables in device memory
// - N, batch: runtime parameters used for dispatch/loops as needed
template <typename T, int N, int batch, bool forward>
__device__ __forceinline__ void
ThunderFFT_kernel_reg(vec2_t<T>* __restrict__ reg, vec2_t<T>* __restrict__ W, void *workspace);

}  // namespace thunderfft


#include "thunderfft/detail/thunderfft_impl.cuh"