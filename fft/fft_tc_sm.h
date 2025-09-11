#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "utils/helper.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


template <typename T>
constexpr const char* type_cstr() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>)   return "float";
    else if constexpr (std::is_same_v<U, double>)  return "double";
    else if constexpr (std::is_same_v<U, int>)     return "int";
    else if constexpr (std::is_same_v<U, __half>)  return "half";
    else if constexpr (std::is_same_v<U, float2>)  return "float2";
    else if constexpr (std::is_same_v<U, __half2>) return "half2";
    else return "unknown";
}

template <uint64_t N>
inline constexpr unsigned LOG2P_builtin = []{
    static_assert(N && (N & (N-1)) == 0, "N must be power of two");
    return __builtin_ctzll(N);    // pow2에서 log2(N)과 동일
}();

template <class T> struct vec2_of;
template <> struct vec2_of<float> { using type = float2; };
template <> struct vec2_of<half>  { using type = half2;  };
template <class T>
using vec2_t = typename vec2_of<std::remove_cv_t<T>>::type;

template <int r, int N> __device__ int reverse_bit_groups(int x) {
    int num_groups = N / r;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (r * i)) & ((1 << r) - 1);
        result |= group << (r * (num_groups - 1 - i));
    }
    return result;
}

__device__ cuFloatComplex W(int index, int N) {
    return make_cuFloatComplex(__cosf(-2 * PI * index / N),
                               __sinf(-2 * PI * index / N));
}

static __device__ __forceinline__
void mma_m16n16k16_fp16_fp16_rowcol(unsigned int d[4], const unsigned int a[4],
                                                   const unsigned int b[4],
                                                   const unsigned int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};\n"
                 : "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[2]), "r"(b[3]),
                   "r"(c[2]), "r"(c[3]));
}

template<typename T, bool inverse>
__device__ __forceinline__ void fill_reg_b(vec2_t<T> _b[], int stride, int k) {
    using T2 = vec2_t<T>;
    int i = (threadIdx.x % 4) ; //& 7;
    int j = (threadIdx.x / 8) ; //& 7;

    //i'th row, j'th col : w_8s ^ ( i (js+k))
    auto b = reinterpret_cast<T*>(_b);

    // auto w1 = W(2*i*(k+stride*j),8*stride);
    // auto w2 = W((2*i+1)*(k+stride*j),8*stride);
    // auto w3 = W(2*i*(k+stride*(j+4)),8*stride);
    // auto w4 = W((2*i+1)*(k+stride*(j+4)),8*stride);

    auto w1 = W((i)*(k+stride*j)+(2*stride * ((threadIdx.x/4)&1)),8*stride);
    auto w2 = W((i+4)*(k+stride*j)+(2*stride * ((threadIdx.x/4)&1)),8*stride);
    auto w3 = (i&1)? make_cuFloatComplex(-w1.x, -w1.y) : make_cuFloatComplex(w1.x, w1.y);
    auto w4 = (i&1)? make_cuFloatComplex(-w2.x, -w2.y) : make_cuFloatComplex(w2.x, w2.y);

    // auto w3 = W((i)*(k+stride*(j+4))+(2*stride * ((threadIdx.x/4)&1)),8*stride);
    // auto w4 = W((i+4)*(k+stride*(j+4))+(2*stride * ((threadIdx.x/4)&1)),8*stride);

    if constexpr (!inverse) {
        b[0] = T(w1.x);
        b[1] = T(-w1.y);
        b[2] = T(w2.x);
        b[3] = T(-w2.y);
        b[4] = T(w3.x);
        b[5] = T(-w3.y);
        b[6] = T(w4.x);
        b[7] = T(-w4.y);
    } else {
        b[0] = T(w1.x);
        b[1] = T(w1.y);
        b[2] = T(w2.x);
        b[3] = T(w2.y);
        b[4] = T(w3.x);
        b[5] = T(w3.y);
        b[6] = T(w4.x);
        b[7] = T(w4.y);
    }
}

template<typename T, unsigned int N, unsigned int radix, bool inverse>
__global__ void kernel_fft_tc_sm(vec2_t<T>* d_data, unsigned int inside_repeats) {
    using T2 = vec2_t<T>; 

    // Tensor core tile: m16n16k16 (A:half, B:half, C/D:float)
    constexpr unsigned int batch            = 16;
    constexpr unsigned int warp_size        = 32;
    constexpr unsigned int ept              = (N * batch) / warp_size;   // elements per thread (if needed)
    constexpr unsigned int element_per_frag = 4;                          // per-lane fragment size for this MMA path

    constexpr unsigned int bank_padding = 9;

    constexpr unsigned int LOG2N = LOG2P_builtin<N>;
    constexpr unsigned int LOG2R = LOG2P_builtin<radix>;

    // (optional) compile-time sanity checks
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    // static_assert((1u << LOG2N) == N, "LOG2N must equal log2(N)"); // 필요하면 활성화

    (void)ept;
    (void)inside_repeats;

    // Dynamic shared memory for this block (typed). Size is provided at kernel launch.
    extern __shared__ void* __smem[];
    auto smem = (T2*)__smem + (N+bank_padding) * batch * threadIdx.y;

    // Per-lane constants/temporaries
    const int laneid = threadIdx.x & 31;

    // Zero fragment for the MMA accumulator C
    T reg_frag_zero[element_per_frag*2];
    for (int t = 0; t < element_per_frag*2; ++t) reg_frag_zero[t] = T(0.0f);

    // --- Load: gmem -> smem (사용자 구현부) ---
    for(int j=0; j<batch; j++) {
        for (int i = threadIdx.x; i < N; i+=warp_size) {
            smem[i+ (N+bank_padding) * j] = d_data[reverse_bit_groups<LOG2R, LOG2N>(i) + N * j +threadIdx.y * N * batch + blockIdx.x * blockDim.y * (N * batch)];
        }
    }
    __syncwarp();

    auto smem_batch0 = smem + (N+bank_padding) * (threadIdx.x / 4);
    auto smem_batch1 = smem + (N+bank_padding) * (threadIdx.x / 4) + (N+bank_padding) * (batch / 2);
    
    T2 reg_frag_a[element_per_frag];
    T2 reg_frag_b[element_per_frag];
    T2 reg_frag_d[element_per_frag];

    for(int iter=0; iter<inside_repeats; iter++) {
        for (unsigned int i = 0; i < LOG2N; i += LOG2R) {
            const int stride = 1 << i;
            #pragma unroll
            for(int jk = 0; jk < N/8; jk++) {
                int j = (jk/stride)*stride*8;
                int k = jk%stride;

                reg_frag_a[0] = smem_batch0[j+k+stride * ((threadIdx.x%4))];
                reg_frag_a[2] = smem_batch0[j+k+stride * ((threadIdx.x%4)+4)];

                reg_frag_a[1] = smem_batch1[j+k+stride * ((threadIdx.x%4))];
                reg_frag_a[3] = smem_batch1[j+k+stride * ((threadIdx.x%4)+4)];

                // --- Fill per-lane fragments from smem (사용자 제공 헬퍼) ---
                fill_reg_b<T, false>(reg_frag_b, stride, k);

                // --- MMA: D = A x B + C ---
                mma_m16n16k16_fp16_fp16_rowcol(
                    (unsigned int*)reg_frag_d, (unsigned int*)reg_frag_a, (unsigned int*)reg_frag_b, (unsigned int*)reg_frag_zero);

                // --- Store result fragment back to smem (사용자 제공 헬퍼) ---

                smem_batch0[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[0];
                smem_batch0[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[2];

                smem_batch1[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[1];
                smem_batch1[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[3];
            }
            __syncwarp();
        }
    }

    // TODO : LOG2N % LOG2R 인 경우 처리

    for(int j=0; j<batch; j++) {
        for (int i = threadIdx.x; i < N; i+=warp_size) {
            d_data[i + N * j +threadIdx.y * N * batch + blockIdx.x * blockDim.y * (N * batch)] = smem[i+ (N+bank_padding) * j];
        }
    }
}

template<typename T, unsigned int N, unsigned int radix>
void fft_tc_sm_perf(vec2_t<T> *d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1000;
    static constexpr unsigned int kernel_runs    = 10;
    static constexpr unsigned int warm_up_runs   = 1;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    int bank_padding = 9;

    // T h_W_64[64];
    // T h_W_4096[4096];

    // auto make_val = [](float re, float im) {
    //     if constexpr (std::is_same_v<T, cuFloatComplex>) {
    //         return make_cuFloatComplex(re, im);
    //     } else if constexpr (std::is_same_v<T, half2>) {
    //         // float -> half2 변환 (round-to-nearest-even)
    //         return __floats2half2_rn(re, im);
    //     } else {
    //         static_assert(!std::is_same_v<T,T>, "Unsupported T");
    //     }
    // };

    // const float PI = 3.14159265358979323846f;

    // for (int i = 0; i < 64; ++i) {
    //     float theta = -2.0f * PI * i / 64.0f;
    //     h_W_64[i] = make_val(cosf(theta), sinf(theta));
    // }

    // for (int i = 0; i < 4096; ++i) {
    //     float theta = -2.0f * PI * i / 4096.0f;
    //     h_W_4096[i] = make_val(cosf(theta), sinf(theta));
    // }

    // T *W_64, *W_4096;
    // CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(T)));
    // CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(T)));
    // CHECK_CUDA(cudaMemcpy(W_64, h_W_64, 64 * sizeof(T),
    //                       cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(T),
    //                       cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaDeviceSynchronize());

    static constexpr unsigned int warp_per_block = 1;

    double elapsed_time_repeat = measure_execution_ms(
        [&](cudaStream_t stream) {
            kernel_fft_tc_sm<T, N, radix, false><<<B / (16 * warp_per_block), dim3(32, warp_per_block), 16*(N+bank_padding)*sizeof(T)*2*warp_per_block, stream>>>(d_data, inside_repeats);
            // assert("4096 half is not supported" && false);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double elapsed_time_repeatx2 = measure_execution_ms(
        [&](cudaStream_t stream) {
            kernel_fft_tc_sm<T, N, radix, false><<<B / (16 * warp_per_block), dim3(32, warp_per_block), 16*(N+bank_padding)*sizeof(T)*2*warp_per_block, stream>>>(d_data, 2 * inside_repeats);
            // assert("4096 half is not supported" && false);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[perf] computation time (type = %s, N = %d, radix = %d): %.8f ms\n", type_cstr<T>(), N, radix, (elapsed_time_repeatx2-elapsed_time_repeat)/inside_repeats);


    double elapsed_time_end_to_end = measure_execution_ms(
        [&](cudaStream_t stream) {
            kernel_fft_tc_sm<T, N, radix, false><<<B / (16 * warp_per_block), dim3(32, warp_per_block), 16*(N+bank_padding)*sizeof(T)*2*warp_per_block, stream>>>(d_data, 1);
            // assert("4096 half is not supported" && false);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("[perf] end-to-end time (type = %s, N = %d, radix = %d): %.8f ms\n", type_cstr<T>(), N, radix, elapsed_time_end_to_end);
}

template<typename T, unsigned int N, unsigned int radix>
void fft_tc_sm_val(vec2_t<T> *d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs    = 1;
    static constexpr unsigned int warm_up_runs   = 0;

    int bank_padding = 9;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    static constexpr unsigned int warp_per_block = 1;

    double elapsed_time_repeat = measure_execution_ms(
        [&](cudaStream_t stream) {
            kernel_fft_tc_sm<T, N, radix, false><<<B / (16 * warp_per_block),32, 16*(N+bank_padding)*sizeof(T)*2*warp_per_block, stream>>>(d_data, inside_repeats);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // printf("computation(once) time (type = %s, N = %d, radix = %d): %.8f ms\n", type_cstr<T>(), N, radix, elapsed_time_repeat);
}