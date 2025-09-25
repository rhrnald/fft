#pragma once
#include "utils.h"

static __device__ __forceinline__ void
mma_m16n16k16_fp16_fp16_rowcol(unsigned int d[4], const unsigned int a[4],
                               const unsigned int b[4],
                               const unsigned int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "r"(c[0]), "r"(c[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};\n"
                 : "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]),
                   "r"(b[3]), "r"(c[2]), "r"(c[3]));
}

static __device__ __forceinline__ void
mma_m16n16k16_tf32_fp32_rowcol(unsigned int d[8], const unsigned int a[8],
                               const unsigned int b[8],
                               const unsigned int c[8]) {

    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[2]),
                   "r"(b[3]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));

    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[4]),
                   "r"(b[5]), "r"(c[4]), "r"(c[5]), "r"(c[6]), "r"(c[7]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
                 : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[6]),
                   "r"(b[7]), "r"(d[4]), "r"(d[5]), "r"(d[6]), "r"(d[7]));

    // __syncthreads();
    // if (blockIdx.x == 0) {
    // printf("(%2d) a : %f %f %f %f %f %f %f %f\n",
    //        threadIdx.x,
    //        __int_as_float(a[0]), __int_as_float(a[1]),
    //        __int_as_float(a[2]), __int_as_float(a[3]),
    //        __int_as_float(a[4]), __int_as_float(a[5]),
    //        __int_as_float(a[6]), __int_as_float(a[7]));

    // printf("(%2d) b : %f %f %f %f %f %f %f %f\n",
    //        threadIdx.x,
    //        __int_as_float(b[0]), __int_as_float(b[1]),
    //        __int_as_float(b[2]), __int_as_float(b[3]),
    //        __int_as_float(b[4]), __int_as_float(b[5]),
    //        __int_as_float(b[6]), __int_as_float(b[7]));
    // printf("(%2d) d : %f %f %f %f %f %f %f %f\n",
    //        threadIdx.x,
    //        __int_as_float(d[0]), __int_as_float(d[1]),
    //        __int_as_float(d[2]), __int_as_float(d[3]),
    //        __int_as_float(d[4]), __int_as_float(d[5]),
    //        __int_as_float(d[6]), __int_as_float(d[7]));
    // }

    // __syncthreads();
}

static __device__ __forceinline__ void
mma_m16n16k16_tf32_fp32_rowcol_new(unsigned int d[8], const unsigned int a[8],
                                   const unsigned int b[8],
                                   const unsigned int c[8]) {
    // C 초기값을 로컬 레지스터 변수로
    unsigned d0 = c[0], d1 = c[1], d2 = c[2], d3 = c[3];

    // (A0..3) x (B0..1) + C0..3 -> D0..3
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) // D/C (read-write)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), // A
                   "r"(b[0]), "r"(b[1])                        // B
    );

    // (A4..7) x (B2..3) + D0..3 -> D0..3
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[2]),
                   "r"(b[3]));

    // 다음 4개 누산기
    unsigned d4 = c[4], d5 = c[5], d6 = c[6], d7 = c[7];

    // (A0..3) x (B4..5) + C4..7 -> D4..7
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(d4), "+r"(d5), "+r"(d6), "+r"(d7)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[4]),
                   "r"(b[5]));

    // (A4..7) x (B6..7) + D4..7 -> D4..7
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(d4), "+r"(d5), "+r"(d6), "+r"(d7)
                 : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[6]),
                   "r"(b[7]));

    // 결과 저장
    d[0] = d0;
    d[1] = d1;
    d[2] = d2;
    d[3] = d3;
    d[4] = d4;
    d[5] = d5;
    d[6] = d6;
    d[7] = d7;
}

template <typename T, int radix, bool inverse>
__device__ __forceinline__ void fill_reg_b_sm(T b[], unsigned int stride,
                                              unsigned int k, unsigned int N);

template <typename T, unsigned int radix, bool inverse>
__device__ void tc_kernel_fft_tc_sm(vec2_t<T> *smem_batch0,
                                    vec2_t<T> *smem_batch1, unsigned int stride,
                                    unsigned int N);

template <typename T, bool inverse>
__device__ void cc_kernel_fft_sm(vec2_t<T> *smem_batch0, vec2_t<T> *smem_batch1,
                                 unsigned int stride, unsigned int N) {
    using T2 = vec2_t<T>;
    T2 tmp[4];
    for (int jk = threadIdx.x % 4; jk < N / 2; jk += 4) {
        int j = (jk / stride) * stride * 2;
        int k = jk % stride;

        tmp[0] = smem_batch0[j + k];
        tmp[1] = smem_batch0[j + k + stride];

        tmp[2] = smem_batch1[j + k];
        tmp[3] = smem_batch1[j + k + stride];

        // w: float2 twiddle
        float2 w = W(k, 2 * stride);
        if constexpr (inverse) {
            w.y = -w.y;
        }

        // ▶ 여기만 교체
        tmp[1] = cmul<T>(tmp[1], w);
        tmp[3] = cmul<T>(tmp[3], w);

        smem_batch0[j + k] = tmp[0] + tmp[1];
        smem_batch0[j + k + stride] = tmp[0] - tmp[1];

        smem_batch1[j + k] = tmp[2] + tmp[3];
        smem_batch1[j + k + stride] = tmp[2] - tmp[3];
    }
}

template <>
__device__ __forceinline__ void
fill_reg_b_sm<half, 8, false>(half b[], unsigned int stride, unsigned int k,
                              unsigned int N) {
    int i = (threadIdx.x % 4); //& 7;
    int j = (threadIdx.x / 8); //& 7;

    // i'th row, j'th col : w_8s ^ ( i (js+k)

    // auto w1 = W(2*i*(k+stride*j),8*stride);
    // auto w2 = W((2*i+1)*(k+stride*j),8*stride);
    // auto w3 = W(2*i*(k+stride*(j+4)),8*stride);
    // auto w4 = W((2*i+1)*(k+stride*(j+4)),8*stride);

    auto w1 = W((i) * (k + stride * j) + (2 * stride * ((threadIdx.x / 4) & 1)),
                8 * stride);
    auto w2 =
        W((i + 4) * (k + stride * j) + (2 * stride * ((threadIdx.x / 4) & 1)),
          8 * stride);
    auto w3 = (i & 1) ? make_float2(-w1.x, -w1.y) : make_float2(w1.x, w1.y);
    auto w4 = (i & 1) ? make_float2(-w2.x, -w2.y) : make_float2(w2.x, w2.y);

    // auto w3 = W((i)*(k+stride*(j+4))+(2*stride *
    // ((threadIdx.x/4)&1)),8*stride); auto w4 =
    // W((i+4)*(k+stride*(j+4))+(2*stride * ((threadIdx.x/4)&1)),8*stride);

    // if constexpr (!inverse) {
    b[0] = half(w1.x);
    b[1] = half(-w1.y);
    b[2] = half(w2.x);
    b[3] = half(-w2.y);
    b[4] = half(w3.x);
    b[5] = half(-w3.y);
    b[6] = half(w4.x);
    b[7] = half(-w4.y);
    // } else {
    //     b[0] = T(w1.x);
    //     b[1] = T(w1.y);
    //     b[2] = T(w2.x);
    //     b[3] = T(w2.y);
    //     b[4] = T(w3.x);
    //     b[5] = T(w3.y);
    //     b[6] = T(w4.x);
    //     b[7] = T(w4.y);
    // }
}

template <>
__device__ void tc_kernel_fft_tc_sm<half, 8, false>(vec2_t<half> *smem_batch0,
                                                    vec2_t<half> *smem_batch1,
                                                    unsigned int stride,
                                                    unsigned int N) {
    // constexpr unsigned int N = 64;
    constexpr unsigned int element_per_frag = 8;
    half reg_frag_zero[element_per_frag];

    for (int t = 0; t < element_per_frag; ++t)
        reg_frag_zero[t] = half(0.0f);

    half2 reg_frag_a[element_per_frag / 2];
    half2 reg_frag_b[element_per_frag / 2];
    half2 reg_frag_d[element_per_frag / 2];

    for (int jk = 0; jk < N / element_per_frag; jk++) {
        int j = (jk / stride) * stride * element_per_frag;
        int k = jk % stride;

        reg_frag_a[0] = smem_batch0[j + k + stride * ((threadIdx.x % 4))];
        reg_frag_a[2] = smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)];

        reg_frag_a[1] = smem_batch1[j + k + stride * ((threadIdx.x % 4))];
        reg_frag_a[3] = smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)];

        // --- Fill per-lane fragments from smem (사용자 제공 헬퍼) ---
        fill_reg_b_sm<half, 8, false>((half *)reg_frag_b, stride, k, 64);

        // --- MMA: D = A x B + C ---
        mma_m16n16k16_fp16_fp16_rowcol(
            (unsigned int *)reg_frag_d, (unsigned int *)reg_frag_a,
            (unsigned int *)reg_frag_b, (unsigned int *)reg_frag_zero);

        // --- Store result fragment back to smem (사용자 제공 헬퍼) ---

        smem_batch0[j + k + stride * ((threadIdx.x % 4))] = reg_frag_d[0];
        smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)] = reg_frag_d[2];

        smem_batch1[j + k + stride * ((threadIdx.x % 4))] = reg_frag_d[1];
        smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)] = reg_frag_d[3];
    }
}

template <>
__device__ __forceinline__ void
fill_reg_b_sm<float, 8, false>(float b[], unsigned int stride, unsigned int k,
                               unsigned int N) {
    int i = (threadIdx.x % 4); //& 7;
    int j = (threadIdx.x / 8); //& 7;

    // i'th row, j'th col : w_8s ^ ( i (js+k))

    // auto w1 = W(2*i*(k+stride*j),8*stride);
    // auto w2 = W((2*i+1)*(k+stride*j),8*stride);
    // auto w3 = W(2*i*(k+stride*(j+4)),8*stride);
    // auto w4 = W((2*i+1)*(k+stride*(j+4)),8*stride);

    auto w1 = W((i) * (k + stride * j) + (2 * stride * ((threadIdx.x / 4) & 1)),
                8 * stride);
    auto w2 =
        W((i + 4) * (k + stride * j) + (2 * stride * ((threadIdx.x / 4) & 1)),
          8 * stride);
    auto w3 = (i & 1) ? make_float2(-w1.x, -w1.y) : make_float2(w1.x, w1.y);
    auto w4 = (i & 1) ? make_float2(-w2.x, -w2.y) : make_float2(w2.x, w2.y);

    // auto w3 = W((i)*(k+stride*(j+4))+(2*stride *
    // ((threadIdx.x/4)&1)),8*stride); auto w4 =
    // W((i+4)*(k+stride*(j+4))+(2*stride * ((threadIdx.x/4)&1)),8*stride);

    // if constexpr (!inverse) {
    b[0] = w1.x;
    b[1] = -w1.y;
    b[2] = w2.x;
    b[3] = -w2.y;
    b[4] = w3.x;
    b[5] = -w3.y;
    b[6] = w4.x;
    b[7] = -w4.y;
    // } else {
    //     b[0] = T(w1.x);
    //     b[1] = T(w1.y);
    //     b[2] = T(w2.x);
    //     b[3] = T(w2.y);
    //     b[4] = T(w3.x);
    //     b[5] = T(w3.y);
    //     b[6] = T(w4.x);
    //     b[7] = T(w4.y);
    // }
}

template <>
__device__ void
tc_kernel_fft_tc_sm<float, 8, false>(float2 *smem_batch0, float2 *smem_batch1,
                                     unsigned int stride, unsigned int N) {
    // constexpr unsigned int N = 64;
    constexpr unsigned int radix = 8;
    constexpr unsigned int element_per_frag =
        8; // batch(16) * radix(8) * 2 / warp_size(32)
    float reg_frag_zero[element_per_frag];

    for (int t = 0; t < element_per_frag; ++t)
        reg_frag_zero[t] = float(0.0f);

    float reg_frag_a[element_per_frag];
    float reg_frag_b[element_per_frag];
    float reg_frag_d[element_per_frag];

    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.y==0 && threadIdx.x==0){
    //     for(int i=0; i< N; i++) printf("(%f,%f) ", smem_batch0[i].x,
    //     smem_batch0[i].y); printf("\n"); for(int i=0; i< N; i++)
    //     printf("(%f,%f) ", smem_batch1[i].x, smem_batch1[i].y); printf("\n");
    // }
    // __syncthreads();

    for (int jk = 0; jk < N / radix; jk++) {
        int j = (jk / stride) * stride * radix;
        int k = jk % stride;

        reg_frag_a[0] = smem_batch0[j + k + stride * ((threadIdx.x % 4))].x;
        reg_frag_a[1] = smem_batch1[j + k + stride * ((threadIdx.x % 4))].x;

        reg_frag_a[2] = smem_batch0[j + k + stride * ((threadIdx.x % 4))].y;
        reg_frag_a[3] = smem_batch1[j + k + stride * ((threadIdx.x % 4))].y;

        reg_frag_a[4] = smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)].x;
        reg_frag_a[5] = smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)].x;

        reg_frag_a[6] = smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)].y;
        reg_frag_a[7] = smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)].y;

        // --- Fill per-lane fragments from smem (사용자 제공 헬퍼) ---
        fill_reg_b_sm<float, 8, false>(reg_frag_b, stride, k, N);

        // --- MMA: D = A x B + C ---
        mma_m16n16k16_tf32_fp32_rowcol(
            (unsigned int *)reg_frag_d, (unsigned int *)reg_frag_a,
            (unsigned int *)reg_frag_b, (unsigned int *)reg_frag_zero);

        // --- Store result fragment back to smem (사용자 제공 헬퍼) ---

        smem_batch0[j + k + stride * ((threadIdx.x % 4))].x = reg_frag_d[0];
        smem_batch1[j + k + stride * ((threadIdx.x % 4))].x = reg_frag_d[2];

        smem_batch0[j + k + stride * ((threadIdx.x % 4))].y = reg_frag_d[1];
        smem_batch1[j + k + stride * ((threadIdx.x % 4))].y = reg_frag_d[3];

        smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)].x = reg_frag_d[4];
        smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)].x = reg_frag_d[6];

        smem_batch0[j + k + stride * ((threadIdx.x % 4) + 4)].y = reg_frag_d[5];
        smem_batch1[j + k + stride * ((threadIdx.x % 4) + 4)].y = reg_frag_d[7];
    }
}

template <typename T, unsigned int N, unsigned int radix, bool inverse>
__global__ void kernel_fft_tc_sm(vec2_t<T> *d_data,
                                 unsigned int inside_repeats) {
    using T2 = vec2_t<T>;

    // Tensor core tile: m16n16k16 (A:half, B:half, C/D:float)
    constexpr unsigned int batch = 16;
    constexpr unsigned int warp_size = 32;

    constexpr unsigned int bank_padding = 9;

    constexpr unsigned int LOG2N = LOG2P_builtin<N>;
    constexpr unsigned int LOG2R = LOG2P_builtin<radix>;

    // Dynamic shared memory for this block (typed). Size is provided at kernel
    // launch.
    extern __shared__ void *smem_data[];
    T2 *smem_data_typed = reinterpret_cast<T2 *>(smem_data);
    T2 *smem = smem_data_typed + (N + bank_padding) * batch * threadIdx.y;

    // --- Load: gmem -> smem (사용자 구현부) ---
    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            smem[i + (N + bank_padding) * j] =
                d_data[reverse_bit_groups<LOG2R, LOG2N>(i) + N * j +
                       (threadIdx.y + blockIdx.x * blockDim.y) * N * batch];
        }
    }
    __syncwarp();

    T2 *smem_batch0 = smem + (N + bank_padding) * (threadIdx.x / 4);
    T2 *smem_batch1 = smem + (N + bank_padding) * (threadIdx.x / 4) +
                      (N + bank_padding) * (batch / 2);

    // Zero fragment for the MMA accumulator C
    for (int iter = 0; iter < inside_repeats; iter++) {
        for (unsigned int i = 0; i < LOG2N / LOG2R * LOG2R; i += LOG2R) {
            unsigned int stride = 1 << i;
            tc_kernel_fft_tc_sm<T, radix, inverse>(smem_batch0, smem_batch1,
                                                   stride, N);
            __syncwarp();
        }
        for (unsigned int i = LOG2N - LOG2N % LOG2R; i < LOG2N; i++) {
            unsigned int stride = 1 << i;
            cc_kernel_fft_sm<T, inverse>(smem_batch0, smem_batch1, stride, N);
            __syncwarp();
        }
    }

    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            d_data[i + N * j + threadIdx.y * N * batch +
                   blockIdx.x * blockDim.y * (N * batch)] =
                smem[i + (N + bank_padding) * j];
        }
    }
}
