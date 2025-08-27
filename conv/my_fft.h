#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// __global__ void
// fft_kernel_radix64_batch16(cuFloatComplex *d_data,
//                            const cuFloatComplex *__restrict__ W_64);
__device__ void fft_kernel_r64_b16(cuFloatComplex *reg,
                                   const cuFloatComplex *__restrict__ W_64);

__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <long long N> void my_fft(cuFloatComplex *d_data) {
    // fft_kernel<<<1, N/2, N * sizeof(cuFloatComplex)>>>(d_data, N);
    // fft_kernel_radix4<<<1, N/4, N * sizeof(cuFloatComplex)>>>(d_data, N);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cuFloatComplex h_W_64[64];
    cuFloatComplex h_W_4096[4096];
    for (int i = 0; i < 64; i++) {
        h_W_64[i] = make_cuFloatComplex(cosf(-2 * M_PI * i / 64),
                                        sinf(-2 * M_PI * i / 64));
    }
    for (int i = 0; i < 4096; i++) {
        h_W_4096[i] = make_cuFloatComplex(cos((-2 * M_PI * i) / 4096.0),
                                          sin((-2 * M_PI * i) / 4096.0));
    }

    cuFloatComplex *W_64, *W_4096;
    CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMemcpy(W_64, h_W_64, 64 * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    // 타이머 시작
    CHECK_CUDA(cudaEventRecord(start));

    // constexpr unsigned int warp_num=1;
    // dim3 grid(32, warp_num);
    // fft_kernel_radix4_matmul<N,warp_num><<<1, grid, N *
    // sizeof(cuFloatComplex)>>>(d_data);

    fft_kernel_radix4096_batch1<<<N / 4096, dim3(32, 4)>>>(d_data, W_4096);

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("my_fft kernel execution time: %.3f ms\n", milliseconds);

    // debug
    cuFloatComplex *tmp = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
    if (!tmp) {
        fprintf(stderr, "Host malloc failed\n");
        return;
    }

    // GPU → CPU 복사
    CHECK_CUDA(cudaMemcpy(tmp, d_data, N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("(%.2f, %.2f) ", tmp[i * 64 + j].x, tmp[i * 64 + j].y);
    //     }
    //     printf("\n");
    // }
}

#include <cmath>
#include <cooperative_groups/memcpy_async.h>
#include <cuComplex.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <math.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#define PI 3.14159265358979323846f

#define TC_M_DEVICE_CONST 16
#define TC_N_DEVICE_CONST 8
#define TC_K_DEVICE_CONST 8

#define RADIX_DEVICE_CONST (TC_K_DEVICE_CONST / 2) // = 4
#define ITER_DEVICE_CONST 3
#define N_DEVICE_CONST 64 // radix^iter
#define BATCH_DEVICE_CONST TC_M_DEVICE_CONST;
#define WARP_SIZE_DEVICE_CONST 32
#define EPT_DEVICE_CONST                                                       \
    (N_DEVICE_CONST * BATCH_DEVICE_CONST /                                     \
     WARP_SIZE_DEVICE_CONST) // element_per_thread

#define N_CONST 4096
#define RADIX_CONST 64
#define RADIX_UNIT_CONST 64
#define BATCH_UNIT_CONST 16
#define WARP_SIZE_CONST 32
#define EPT_CONST (RADIX_CONST * BATCH_UNIT_CONST / WARP_SIZE_CONST)
#define NUM_WARP_CONST 4

__device__ cuFloatComplex W(int index, int N) {
    return make_cuFloatComplex(__cosf(-2 * PI * index / N),
                               __sinf(-2 * PI * index / N));
}

template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

__device__ int reverse_2bit_groups_6(int x) {
    return ((x & 3) << 4) | (((x >> 2) & 3) << 2) | ((x >> 4) & 3);
}

__device__ int reverse_2bit_groups_5(int x) {
    return ((x & 3) << 3) | (((x >> 2) & 3) << 1) | ((x >> 4) & 1);
}
__device__ int reverse_2bit_groups_5_221(int x) {
    return ((x & 1) << 4) | (((x >> 1) & 3) << 2) | ((x >> 3) & 3);
}

template <bool inverse>
__device__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_64) {
    // b = [ w^ (i+i_perm) ( k + N(j+j_perm)) ] ^ T

    // register mapping
    // 0 4 8   ...   28
    // 1 5 9
    // 2 6 10
    // 3 7 11  ...   31
    int i = (threadIdx.x / 8 - i_perm) & 3;
    int j = (threadIdx.x % 4 - j_perm) & 3;

    // if(i==j) {
    //     if((threadIdx.x/4) & 1) {
    //         b[0] = 0.0f;
    //         b[1] = 1.0f;
    //     } else {
    //         b[0] = 1.0f;
    //         b[1] = 0.0f;
    //     }
    // } else {
    //     b[0]=0.0f;
    //     b[1]=0.0f;
    // }
    // return;

    // auto w = W(j*(k+stride*i),4*stride);
    int index =
        (1 << (4 - stride_log2)) * ((j * (k + stride * i)) & (4 * stride - 1));

    cuFloatComplex w = W_64[index];
    // cuFloatComplex w = W(index, 64);

    if (!inverse) {
        if ((threadIdx.x / 4) & 1) {
            b[0] = w.y;
            b[1] = w.x;
        } else {
            b[0] = w.x;
            b[1] = -w.y;
        }
    } else {
        if ((threadIdx.x / 4) & 1) {
            b[0] = -w.y;
            b[1] = w.x;
        } else {
            b[0] = w.x;
            b[1] = w.y;
        }
    }
}

static __device__ void mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4],
                                                   const float b[2],
                                                   const float c[4]) {
    // __syncwarp();
    auto a0 = __float_as_uint(a[0]);
    auto a1 = __float_as_uint(a[1]);
    auto a2 = __float_as_uint(a[2]);
    auto a3 = __float_as_uint(a[3]);
    auto b0 = __float_as_uint(b[0]);
    auto b1 = __float_as_uint(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "      // D (also C)
                 "{%4, %5, %6, %7}, "      // A (tf32 in .b32 regs)
                 "{%8, %9}, "              // B (tf32 in .b32 regs)
                 "{%10, %11, %12, %13};\n" // C
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <typename T>
__device__ void permute_radix4(T &a, T &b, T &c, T &d, int pattern) {
    T t0 = a, t1 = b, t2 = c, t3 = d;

    switch (pattern & 3) {
    // {0,3,2,1}
    case 0:
        a = t0;
        b = t3;
        c = t2;
        d = t1;
        break;
    // {1,0,3,2}
    case 1:
        a = t1;
        b = t0;
        c = t3;
        d = t2;
        break;
    // {2,1,0,3}
    case 2:
        a = t2;
        b = t1;
        c = t0;
        d = t3;
        break;
    // {3,2,1,0}
    default:
        a = t3;
        b = t2;
        c = t1;
        d = t0;
        break;
    }
    // T tmp[4] = {a,b,c,d};
    // a=tmp[pattern];
    // b=tmp[(pattern-1)&3];
    // c=tmp[(pattern-2)&3];
    // d=tmp[(pattern-3)&3];
}

__device__ void swap_r2c(float2 &z1, float2 &z2, float2 w) {
    float2 tmp1 = make_float2(z1.x + z2.x, z1.y - z2.y);
    float2 tmp2 = make_float2(z1.x - z2.x, z1.y + z2.y);
    tmp2 =
        make_float2(tmp2.x * w.x - tmp2.y * w.y, tmp2.x * w.y + tmp2.y * w.x);
    z1 = make_float2((tmp1.x + tmp2.y) / 2, (tmp1.y - tmp2.x) / 2);
    z2 = make_float2((tmp1.x - tmp2.y) / 2, (-tmp1.y - tmp2.x) / 2);
}

__device__ void swap_c2r(float2 &z1, float2 &z2, float2 w) {
    float2 Ek = make_float2(z1.x + z2.x, z1.y - z2.y);
    float2 Ok = make_float2(z1.x - z2.x, z1.y + z2.y);
    Ok = make_float2(Ok.x * w.x + Ok.y * w.y, -Ok.x * w.y + Ok.y * w.x);
    z1 = make_float2((Ek.x - Ok.y) / 2, (Ek.y + Ok.x) / 2);
    z2 = make_float2((Ek.x + Ok.y) / 2, (-Ek.y + Ok.x) / 2);
}

__device__ void pm(float2 &z1, float2 &z2) {
    z1 = make_float2((z1.x + z2.x) * 2, (z1.y + z2.y) * 2);
    z2 = make_float2(z1.x - 4 * z2.x, z1.y - 4 * z2.y);
}
__device__ void fft_r2c_kernel_r64_b16(cuFloatComplex *reg,
                                       const cuFloatComplex *__restrict__ W_64,
                                       float2 *smem) {
    // Tensor core shape
    // constexpr int m=16;
    // constexpr int n=8;
    // constexpr int k=8;

    // constexpr int radix = k/2; // = 4
    // constexpr int iter = 3;
    // constexpr int N = 64; // radix^iter
    // constexpr int batch=m;
    // constexpr int warp_size=32;
    // constexpr int ept=N * batch / warp_size; // element_per_thread

    // Registers for mma : d = a * b + zero;

    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST / 2; j++) {
        float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];

        reg_frag_a[0] = reg[j].x;
        reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x;
        reg_frag_a[2] = reg[j].y;
        reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y;

        // w = w_4stride
        // b = [ w^ i ( k + Nj) ] ^ T
        int j_perm = 0;
        int i_perm = j & (RADIX_DEVICE_CONST - 1);
        int k = 0;

        fill_reg_b<false>(reg_frag_b, 0, 1, i_perm, j_perm, k, W_64);
        // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
        // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

        mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                    reg_frag_zero);

        reg[j].x = reg_frag_d[0];
        reg[j].y = reg_frag_d[1];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x = reg_frag_d[2];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y = reg_frag_d[3];
    }

    for (int j = 0; j < 16; j += 4) {
        // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
        // t0 t1 t2 t3
        // 0  1  2  3       0  4  8  12
        // 7  4  5  6       13 1  5  9
        // 10 11 8  9	->  10 14 2  6
        // 13 14 15 12		7  11 15 3
        permute_radix4(reg[j], reg[j + 1], reg[j + 2], reg[j + 3], laneid & 3);
    }

    for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST / 2; j++) {
        float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];

        reg_frag_a[0] = reg[j].x;
        reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x;
        reg_frag_a[2] = reg[j].y;
        reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y;

        // w = w_4stride
        // b = [ w^ i ( k + Nj) ] ^ T
        int j_perm = j & (RADIX_DEVICE_CONST - 1);
        int i_perm = 0;
        int k = j & 3;

        fill_reg_b<false>(reg_frag_b, 2, 4, i_perm, j_perm, k, W_64);
        // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
        // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

        mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                    reg_frag_zero);

        reg[j].x = reg_frag_d[0];
        reg[j].y = reg_frag_d[1];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x = reg_frag_d[2];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y = reg_frag_d[3];
    }

    for (int i = 0; i < 4; i++) {
        float2 w_tmp = W_64[(i + (lane_id % 4) * 4) * 2];
        reg[i + 4] =
            make_float2(reg[i + 4].x * w_tmp.x - reg[i + 4].y * w_tmp.y,
                        reg[i + 4].x * w_tmp.y + reg[i + 4].y * w_tmp.x);
        reg[i + 12] =
            make_float2(reg[i + 12].x * w_tmp.x - reg[i + 12].y * w_tmp.y,
                        reg[i + 12].x * w_tmp.y + reg[i + 12].y * w_tmp.x);

        float2 tmp1 = reg[i], tmp2 = reg[i + 4], tmp3 = reg[i + 8],
               tmp4 = reg[i + 12];

        reg[i] = make_float2(tmp1.x + tmp2.x, tmp1.y + tmp2.y);
        reg[i + 4] = make_float2(tmp1.x - tmp2.x, tmp1.y - tmp2.y);
        reg[i + 8] = make_float2(tmp3.x + tmp4.x, tmp3.y + tmp4.y);
        reg[i + 12] = make_float2(tmp3.x - tmp4.x, tmp3.y - tmp4.y);
    }

    for (int i = 0; i < 8; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = 16 * ((i % 8) / 4) + (i % 4) + lane_id % 4 * 4;
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i];
        smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i + 8];
    }
    __syncwarp();

    for (int i = 0; i < 4; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = i + lane_id % 4 * 4;
        reg[i] = smem[row * (RADIX_UNIT_CONST / 2 + 1) + col];
        reg[i + 4] =
            smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16];

        row += 8;
        reg[i + 8] = smem[row * (RADIX_UNIT_CONST / 2 + 1) + col];
        reg[i + 12] =
            smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16];

        if (col) {
            swap_r2c(reg[i], reg[i + 4], W_64[col]);
            swap_r2c(reg[i + 8], reg[i + 12], W_64[col]);
        } else {
            reg[0] = make_float2(reg[0].x + reg[0].y, reg[0].x - reg[0].y);
            reg[4].y = -reg[4].y;

            reg[8] = make_float2(reg[8].x + reg[8].y, reg[8].x - reg[8].y);
            reg[12].y = -reg[12].y;
        }
    }
    __syncwarp();

    for (int i = 0; i < 4; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = i + lane_id % 4 * 4;
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i];
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16] =
            reg[i + 4];
        row += 8;
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i + 8];
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16] =
            reg[i + 12];
    }
}

__device__ void fft_c2r_kernel_r64_b16(cuFloatComplex *reg,
                                       const cuFloatComplex *__restrict__ W_64,
                                       float2 *smem) {
    // Tensor core shape
    // constexpr int m=16;
    // constexpr int n=8;
    // constexpr int k=8;

    // constexpr int radix = k/2; // = 4
    // constexpr int iter = 3;
    // constexpr int N = 64; // radix^iter
    // constexpr int batch=m;
    // constexpr int warp_size=32;
    // constexpr int ept=N * batch / warp_size; // element_per_thread

    // Registers for mma : d = a * b + zero;

    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    for (int i = 0; i < 4; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = i + lane_id % 4 * 4;
        reg[i] = smem[row * (RADIX_UNIT_CONST / 2 + 1) + col];
        reg[i + 4] =
            smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16];
        reg[i + 8] = smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + col];
        reg[i + 12] =
            smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16];
    }

    __syncwarp();

    for (int i = 0; i < 4; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = i + lane_id % 4 * 4;

        if (col) {
            swap_c2r(reg[i], reg[i + 4], W_64[col]);
            swap_c2r(reg[i + 8], reg[i + 12], W_64[col]);
        } else {
            reg[0] = make_float2((reg[0].x + reg[0].y) / 2,
                                 (reg[0].x - reg[0].y) / 2);
            reg[4].y = -reg[4].y;

            reg[8] = make_float2((reg[8].x + reg[8].y) / 2,
                                 (reg[8].x - reg[8].y) / 2);
            reg[12].y = -reg[12].y;
        }

        smem[row * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i];
        smem[row * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16] =
            reg[i + 4];
        smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + col] = reg[i + 8];
        smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + (32 - col) % 16 + 16] =
            reg[i + 12];
    }
    __syncwarp();

    for (int i = 0; i < 8; i++) {
        int row = lane_id / 4 + warp_id * 16;
        int col = ((i % 8) / 2) + (i % 2) * 16 + lane_id % 4 * 4;
        reg[i] = smem[row * (RADIX_UNIT_CONST / 2 + 1) + col];
        reg[i + 8] = smem[(row + 8) * (RADIX_UNIT_CONST / 2 + 1) + col];
        // reg[i]= make_float2(row*(RADIX_UNIT_CONST/2+1) + col,0);
        // reg[i+8] = make_float2((row+8)*(RADIX_UNIT_CONST/2+1) + col,0);
    }
    for (int i = 0; i < 16; i += 2) {
        pm(reg[i], reg[i + 1]);
    }

    for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST / 2; j++) {
        float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];

        reg_frag_a[0] = reg[j].x;
        reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x;
        reg_frag_a[2] = reg[j].y;
        reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y;

        // w = w_4stride
        // b = [ w^ i ( k + Nj) ] ^ T
        int j_perm = 0;
        int i_perm = (j / 2) % RADIX_DEVICE_CONST;
        int k = j % 2;

        fill_reg_b<true>(reg_frag_b, 1, 2, i_perm, j_perm, k, W_64);
        // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
        // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

        mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                    reg_frag_zero);

        reg[j].x = reg_frag_d[0];
        reg[j].y = reg_frag_d[1];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x = reg_frag_d[2];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y = reg_frag_d[3];
    }

    for (int j = 0; j < 16; j += 8) {
        for (int k = 0; k < 2; k++) {
            permute_radix4(reg[k + j], reg[k + j + 2], reg[k + j + 4],
                           reg[k + j + 6], laneid & 3);
        }
    }
    for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST / 2; j++) {
        float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];
        float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                         WARP_SIZE_DEVICE_CONST];

        reg_frag_a[0] = reg[j].x;
        reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x;
        reg_frag_a[2] = reg[j].y;
        reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y;

        // w = w_4stride
        // b = [ w^ i ( k + Nj) ] ^ T
        int j_perm = (j / 2) % RADIX_DEVICE_CONST;
        int i_perm = 0;
        int k = j % 8;

        fill_reg_b<true>(reg_frag_b, 3, 8, i_perm, j_perm, k, W_64);
        // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
        // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

        mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                    reg_frag_zero);

        reg[j].x = reg_frag_d[0];
        reg[j].y = reg_frag_d[1];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].x = reg_frag_d[2];
        reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST / 2].y = reg_frag_d[3];
    }
}

// in-place device kernel
template <bool inverse>
__device__ void fft_kernel_r64_b16(cuFloatComplex *reg,
                                   const cuFloatComplex *__restrict__ W_64) {
    // Tensor core shape
    // constexpr int m=16;
    // constexpr int n=8;
    // constexpr int k=8;

    // constexpr int radix = k/2; // = 4
    // constexpr int iter = 3;
    // constexpr int N = 64; // radix^iter
    // constexpr int batch=m;
    // constexpr int warp_size=32;
    // constexpr int ept=N * batch / warp_size; // element_per_thread

    // Registers for mma : d = a * b + zero;

    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;

    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];

            reg_frag_a[0] = reg[j].x;
            reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x;
            reg_frag_a[2] = reg[j].y;
            reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y;

            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b<inverse>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                W_64);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);

            reg[j].x = reg_frag_d[0];
            reg[j].y = reg_frag_d[1];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x = reg_frag_d[2];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y = reg_frag_d[3];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int j = 0; j < 32; j += 4 * stride) {
                for (int k = 0; k < stride; k++) {
                    // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                    // t0 t1 t2 t3
                    // 0  1  2  3       0  4  8  12
                    // 7  4  5  6       13 1  5  9
                    // 10 11 8  9	->  10 14 2  6
                    // 13 14 15 12		7  11 15 3
                    permute_radix4(reg[k + j], reg[k + j + stride],
                                   reg[k + j + stride * 2],
                                   reg[k + j + stride * 3], laneid & 3);
                }
            }
        }
    }
}
