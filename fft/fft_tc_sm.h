#pragma once
#include "utils.h"

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
__device__ void core_kernel_fft_tc_sm(vec2_t<T>* smem_batch0, vec2_t<T>* smem_batch1, unsigned int stride);

template<>
__device__ void core_kernel_fft_tc_sm<half, 64, 8, false>(vec2_t<half>* smem_batch0, vec2_t<half>* smem_batch1, unsigned int stride) {
    constexpr unsigned int element_per_frag = 4;
    half reg_frag_zero[element_per_frag*2];

    for (int t = 0; t < element_per_frag*2; ++t) reg_frag_zero[t] = half(0.0f);

    half2 reg_frag_a[element_per_frag];
    half2 reg_frag_b[element_per_frag];
    half2 reg_frag_d[element_per_frag];

    for(int jk = 0; jk < 8; jk++) {
        int j = (jk/stride)*stride*8;
        int k = jk%stride;

        reg_frag_a[0] = smem_batch0[j+k+stride * ((threadIdx.x%4))];
        reg_frag_a[2] = smem_batch0[j+k+stride * ((threadIdx.x%4)+4)];

        reg_frag_a[1] = smem_batch1[j+k+stride * ((threadIdx.x%4))];
        reg_frag_a[3] = smem_batch1[j+k+stride * ((threadIdx.x%4)+4)];

        // --- Fill per-lane fragments from smem (사용자 제공 헬퍼) ---
        fill_reg_b<half, false>(reg_frag_b, stride, k);

        // --- MMA: D = A x B + C ---
        mma_m16n16k16_fp16_fp16_rowcol(
            (unsigned int*)reg_frag_d, (unsigned int*)reg_frag_a, (unsigned int*)reg_frag_b, (unsigned int*)reg_frag_zero);

        // --- Store result fragment back to smem (사용자 제공 헬퍼) ---

        smem_batch0[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[0];
        smem_batch0[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[2];

        smem_batch1[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[1];
        smem_batch1[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[3];
    }
}
template<>
__device__ void core_kernel_fft_tc_sm<float, 64, 8, false>(vec2_t<float>* smem_batch0, vec2_t<float>* smem_batch1, unsigned int stride) {}


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

    // --- Load: gmem -> smem (사용자 구현부) ---
    for(int j=0; j<batch; j++) {
        for (int i = threadIdx.x; i < N; i+=warp_size) {
            smem[i+ (N+bank_padding) * j] = d_data[reverse_bit_groups<LOG2R, LOG2N>(i) + N * j +threadIdx.y * N * batch + blockIdx.x * blockDim.y * (N * batch)];
        }
    }
    __syncwarp();

    auto smem_batch0 = smem + (N+bank_padding) * (threadIdx.x / 4);
    auto smem_batch1 = smem + (N+bank_padding) * (threadIdx.x / 4) + (N+bank_padding) * (batch / 2);
    
    // Zero fragment for the MMA accumulator C
    for(int iter=0; iter<inside_repeats; iter++) {
        for (unsigned int i = 0; i < LOG2N; i += LOG2R) {
            unsigned int stride = 1 << i;
            core_kernel_fft_tc_sm<T, N, radix, inverse>(smem_batch0, smem_batch1, stride);
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
