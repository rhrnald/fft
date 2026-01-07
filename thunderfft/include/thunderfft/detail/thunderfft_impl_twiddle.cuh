// thunderfft_impl_twiddle.cuh
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <vector>

#include "util_cuda.h"

namespace thunderfft {

// ============================================================
// Compile-time constants
// ============================================================
constexpr int WARP_SIZE      = 32;
constexpr int UNIT_ENTRIES   = 28;
constexpr int UNIT_W_SIZE    = UNIT_ENTRIES * WARP_SIZE * 2;

// ============================================================
// Host-side twiddle cache (N-dependent)
// ============================================================
template <typename T>
struct TwiddleCache {
  static T*  dW;
  static int N_cached;
};

template <typename T> T*  TwiddleCache<T>::dW       = nullptr;
template <typename T> int TwiddleCache<T>::N_cached = 0;

// ============================================================
// Device-side symbols
// ============================================================
namespace detail {

// N-dependent twiddle
__device__ __constant__ const float* d_twiddle_f   = nullptr;
__device__ __constant__ int          d_twiddle_N_f = 0;

__device__ __constant__ const half*  d_twiddle_h   = nullptr;
__device__ __constant__ int          d_twiddle_N_h = 0;

// N-independent unit twiddle (lane-dependent!)
__device__ __constant__ half2  d_unit_twiddle_half2[UNIT_W_SIZE];
__device__ __constant__ float2 d_unit_twiddle_float2[UNIT_W_SIZE];

} // namespace detail

// ============================================================
// N-dependent twiddle builders
// ============================================================
inline float* preprocess_W_float(int N) {
  if (N <= 0) return nullptr;

  std::vector<float> h(N);
  const double scale = 2.0 * M_PI / double(N);
  for (int k = 0; k < N; ++k) h[k] = float(std::cos(scale * k));

  float* d = nullptr;
  tf_check_cuda(cudaMalloc(&d, sizeof(float) * N), "cudaMalloc dW (float)");
  tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(float) * N,
                            cudaMemcpyHostToDevice),
                "cudaMemcpy dW (float)");
  return d;
}

inline half* preprocess_W_half(int N) {
  if (N <= 0) return nullptr;

  std::vector<half> h(N);
  const double scale = 2.0 * M_PI / double(N);
  for (int k = 0; k < N; ++k)
    h[k] = __float2half(float(std::cos(scale * k)));

  half* d = nullptr;
  tf_check_cuda(cudaMalloc(&d, sizeof(half) * N), "cudaMalloc dW (half)");
  tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(half) * N,
                            cudaMemcpyHostToDevice),
                "cudaMemcpy dW (half)");
  return d;
}

// ============================================================
// CPU-side unit twiddle builder (lane-aware, N-independent)
// ============================================================
template <typename PairT>
inline PairT make_pair(float a, float b);

template <>
inline half2 make_pair<half2>(float a, float b) {
  return make_half2(__float2half(a), __float2half(b));
}

template <>
inline float2 make_pair<float2>(float a, float b) {
  return make_float2(a, b);
}

template <typename PairT>
void build_unit_twiddle_host(std::vector<PairT>& W) {

    // -----------------------------
    // local helper (TEMPORARY)
    // -----------------------------
    auto f = [](int x) {
        return (x & 1) * 4 + (x >> 1);
    };

    W.resize(UNIT_W_SIZE);

    constexpr int n     = 64;
    constexpr int radix = 4;

    int entry = 0;

    for (int stage = 0; stage < 3; ++stage) {
        const int stride = 1 << (stage << 1);

        for (int _j = 0; _j < n / radix; ++_j) {
            if (_j % (1 << (2 - stage)) != 0) continue;

            int j = (_j % 4) * 4 + _j / 4;

            const int j_perm =
                (stride >= radix)
                  ? ((j / (stride / radix)) / 2 * 2) % radix
                  : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            for (int lane = 0; lane < 32; ++lane) {
                for (int dir = 0; dir < 2; ++dir) {
                    const bool forward = (dir == 0);

                    int i0 = lane / 4;
                    int i1 = lane / 4;
                    int j0 = (lane % 4) * 2;
                    int j1 = (lane % 4) * 2 + 1;

                    if (stage == 0) {
                        j0 = f(j0);
                        j1 = f(j1);
                    }
                    if (stage == 2) {
                        i0 = f(i0);
                        i1 = f(i1);
                    }

                    i0 ^= i_perm;  i1 ^= i_perm;
                    j0 ^= j_perm;  j1 ^= j_perm;

                    i0 = (i0 % 4) * 2 + i0 / 4;
                    i1 = (i1 % 4) * 2 + i1 / 4;
                    j0 = (j0 % 4) * 2 + j0 / 4;
                    j1 = (j1 % 4) * 2 + j1 / 4;

                    const int index1 =
                        (j0 / 2) * (k + stride * (i0 / 2)) +
                        stride * (i0 & 1) - stride * (j0 & 1);
                    const int index2 =
                        (j1 / 2) * (k + stride * (i1 / 2)) +
                        stride * (i1 & 1) - stride * (j1 & 1);

                    float c1 = std::cos(2.0f * M_PI * index1 / float(4 * stride));
                    float c2 = std::cos(2.0f * M_PI * index2 / float(4 * stride));

                    if (!forward) {
                        if ((i0 + j0) & 1) c1 = -c1;
                        if ((i1 + j1) & 1) c2 = -c2;
                    }

                    const int idx =
                        ((dir * UNIT_ENTRIES + entry) * 32 + lane);

                    W[idx] = make_pair<PairT>(c1, c2);
                }
            }

            ++entry;
        }
    }

    assert(entry == UNIT_ENTRIES);
}

// ============================================================
// Public API
// ============================================================
template <typename T>
inline void ThunderFFTInitialize(int N) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                "ThunderFFTInitialize<T>: supported T are float and half only");

  // ---------- unit twiddle (once, lane-aware) ----------
  if constexpr (std::is_same_v<T, half>) {
    static bool unit_inited_half = false;
    if (!unit_inited_half) {
      std::vector<half2> h_unit;
      build_unit_twiddle_host(h_unit);

      tf_check_cuda(cudaMemcpyToSymbol(
          detail::d_unit_twiddle_half2,
          h_unit.data(),
          sizeof(half2) * UNIT_W_SIZE),
          "MemcpyToSymbol(d_unit_twiddle_half2)");

      unit_inited_half = true;
    }
  } else {
    static bool unit_inited_float = false;
    if (!unit_inited_float) {
      std::vector<float2> h_unit;
      build_unit_twiddle_host(h_unit);

      tf_check_cuda(cudaMemcpyToSymbol(
          detail::d_unit_twiddle_float2,
          h_unit.data(),
          sizeof(float2) * UNIT_W_SIZE),
          "MemcpyToSymbol(d_unit_twiddle_float2)");

      unit_inited_float = true;
    }
  }

  // ---------- N-dependent twiddle ----------
  if (TwiddleCache<T>::dW && TwiddleCache<T>::N_cached == N) return;

  if (TwiddleCache<T>::dW) {
    cudaFree(TwiddleCache<T>::dW);
    TwiddleCache<T>::dW = nullptr;
    TwiddleCache<T>::N_cached = 0;
  }

  if constexpr (std::is_same_v<T, float>) {
    TwiddleCache<float>::dW = preprocess_W_float(N);
    TwiddleCache<float>::N_cached = N;

    cudaMemcpyToSymbol(detail::d_twiddle_f,
                       &TwiddleCache<float>::dW,
                       sizeof(float*));
    cudaMemcpyToSymbol(detail::d_twiddle_N_f, &N, sizeof(int));
  } else {
    TwiddleCache<half>::dW = preprocess_W_half(N);
    TwiddleCache<half>::N_cached = N;

    cudaMemcpyToSymbol(detail::d_twiddle_h,
                       &TwiddleCache<half>::dW,
                       sizeof(half*));
    cudaMemcpyToSymbol(detail::d_twiddle_N_h, &N, sizeof(int));
  }
}

// ============================================================
// Device-side accessors
// ============================================================
template <typename T>
__device__ __forceinline__ const T* ThunderFFT_get_twiddle();

template <>
__device__ __forceinline__ const float* ThunderFFT_get_twiddle<float>() {
  return detail::d_twiddle_f;
}

template <>
__device__ __forceinline__ const half* ThunderFFT_get_twiddle<half>() {
  return detail::d_twiddle_h;
}

__device__ __forceinline__ half2*
ThunderFFT_get_unit_twiddle_half2() {
  return detail::d_unit_twiddle_half2;
}

__device__ __forceinline__ float2*
ThunderFFT_get_unit_twiddle_float2() {
  return detail::d_unit_twiddle_float2;
}

template <typename T>
inline void ThunderFFTFinalize() {
  if (TwiddleCache<T>::dW) {
    cudaFree(TwiddleCache<T>::dW);
    TwiddleCache<T>::dW = nullptr;
    TwiddleCache<T>::N_cached = 0;
  }
}


} // namespace thunderfft
