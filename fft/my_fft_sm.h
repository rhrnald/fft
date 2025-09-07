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

template <int N> __device__ int reverse_3bit_groups(int x) {
    int num_groups = N / 3;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (3 * i)) & 0b111;
        result |= group << (3 * (num_groups - 1 - i));
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
                                                   const unsigned int c[4], void* smem) {
    // __syncthreads();
    // if(threadIdx.y==0) {
    //     int idx=threadIdx.x;

    //     printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)a)[0].x), __half2float(((half2*)a)[0].y), 
    //                                                 __half2float(((half2*)a)[1].x), __half2float(((half2*)a)[1].y), 
    //                                                 __half2float(((half2*)a)[2].x), __half2float(((half2*)a)[2].y), 
    //                                                 __half2float(((half2*)a)[3].x), __half2float(((half2*)a)[3].y));
        
    //     printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)b)[0].x), __half2float(((half2*)b)[0].y), 
    //                                                 __half2float(((half2*)b)[1].x), __half2float(((half2*)b)[1].y), 
    //                                                 __half2float(((half2*)b)[2].x), __half2float(((half2*)b)[2].y), 
    //                                                 __half2float(((half2*)b)[3].x), __half2float(((half2*)b)[3].y));
                                                    
    //     printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)c)[0].x), __half2float(((half2*)c)[0].y), 
    //                                                 __half2float(((half2*)c)[1].x), __half2float(((half2*)c)[1].y), 
    //                                                 __half2float(((half2*)c)[2].x), __half2float(((half2*)c)[2].y), 
    //                                                 __half2float(((half2*)c)[3].x), __half2float(((half2*)c)[3].y));
    // }
    // __syncthreads();

    /*if(threadIdx.y==0) {
        int idx=threadIdx.x;

        // printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)a)[0].x), __half2float(((half2*)a)[0].y), 
        //                                             __half2float(((half2*)a)[1].x), __half2float(((half2*)a)[1].y), 
        //                                             __half2float(((half2*)a)[2].x), __half2float(((half2*)a)[2].y), 
        //                                             __half2float(((half2*)a)[3].x), __half2float(((half2*)a)[3].y));

        // ((unsigned int*)smem)[(idx/4) * 8 + (idx%4)*2] = a[0]; 
        // ((unsigned int*)smem)[(idx/4) * 8 + (idx%4)*2+1] = a[1];
        // ((unsigned int*)smem)[(idx/4+8) * 8 + (idx%4)*2] = a[2];
        // ((unsigned int*)smem)[(idx/4+8) * 8 + (idx%4)*2+1] = a[3];

        // if(threadIdx.x==0 && threadIdx.y==0) {
        //     for(int i=0; i<16; i++) {
        //         for(int j=0; j<8; j++) {
        //             printf("(%f,%f)", __half2float(((half2*)smem)[i*8+j].x), __half2float(((half2*)smem)[i*8+j].y));
        //         }
        //     printf("\n");
        //     }
        //     printf("------------------\n");
        // }
        printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)b)[0].x), __half2float(((half2*)b)[0].y), 
                                                    __half2float(((half2*)b)[1].x), __half2float(((half2*)b)[1].y), 
                                                    __half2float(((half2*)b)[2].x), __half2float(((half2*)b)[2].y), 
                                                    __half2float(((half2*)b)[3].x), __half2float(((half2*)b)[3].y));

        ((unsigned int*)smem)[(idx/4) * 8 + (idx%4)*2] = b[0]; 
        ((unsigned int*)smem)[(idx/4) * 8 + (idx%4)*2+1] = b[1];
        ((unsigned int*)smem)[(idx/4+8) * 8 + (idx%4)*2] = b[2];
        ((unsigned int*)smem)[(idx/4+8) * 8 + (idx%4)*2+1] = b[3];

        if(threadIdx.x==0 && threadIdx.y==0) {
            for(int i=0; i<16; i++) {
                for(int j=0; j<8; j++) {
                    printf("(%f,%f)", __half2float(((half2*)smem)[i*8+j].x), __half2float(((half2*)smem)[i*8+j].y));
                }
            printf("\n");
            }
            printf("------------------\n");
        }
    }
    __syncthreads();*/

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

    // __syncthreads();
    // if(threadIdx.y==0) {
    //     int idx=threadIdx.x;

    //     printf("%d:%f %f %f %f %f %f %f %f\n", idx, __half2float(((half2*)d)[0].x), __half2float(((half2*)d)[0].y), 
    //                                                 __half2float(((half2*)d)[1].x), __half2float(((half2*)d)[1].y), 
    //                                                 __half2float(((half2*)d)[2].x), __half2float(((half2*)d)[2].y), 
    //                                                 __half2float(((half2*)d)[3].x), __half2float(((half2*)d)[3].y));

    // }
    // __syncthreads();
}

template<bool inverse>
__device__ __forceinline__ void fill_reg_b(half2 _b[], int stride, int k) {
    int i = (threadIdx.x % 4) ; //& 7;
    int j = (threadIdx.x / 8) ; //& 7;

    //i'th row, j'th col : w_8s ^ ( i (js+k))
    auto b = reinterpret_cast<half*>(_b);

    // auto w1 = W(2*i*(k+stride*j),8*stride);
    // auto w2 = W((2*i+1)*(k+stride*j),8*stride);
    // auto w3 = W(2*i*(k+stride*(j+4)),8*stride);
    // auto w4 = W((2*i+1)*(k+stride*(j+4)),8*stride);

    
    auto w1 = W((i)*(k+stride*j),8*stride);
    auto w2 = W((i+4)*(k+stride*j),8*stride);
    auto w3 = W((i)*(k+stride*(j+4)),8*stride);
    auto w4 = W((i+4)*(k+stride*(j+4)),8*stride);
    // printf("threadIdx.x: %d, i: %d, j: %d, 2*i*(k+stride*j): %d, w1: (%f, %f), w2: (%f, %f)\n", threadIdx.x, i, j, 2*i*(k+stride*j), w1.x,w1.y, w2.x,w2.y);

    if constexpr (!inverse) {
        if ((threadIdx.x / 4) & 1) {
            b[0] = __float2half(w1.y);
            b[1] = __float2half(w1.x);
            b[2] = __float2half(w2.y);
            b[3] = __float2half(w2.x);
            b[4] = __float2half(w3.y);
            b[5] = __float2half(w3.x);
            b[6] = __float2half(w4.y);
            b[7] = __float2half(w4.x);
        } else {
            b[0] = __float2half(w1.x);
            b[1] = __float2half(-w1.y);
            b[2] = __float2half(w2.x);
            b[3] = __float2half(-w2.y);
            b[4] = __float2half(w3.x);
            b[5] = __float2half(-w3.y);
            b[6] = __float2half(w4.x);
            b[7] = __float2half(-w4.y);
        }
    } else {
        if ((threadIdx.x / 4) & 1) {
            b[0] = __float2half(-w1.y);
            b[1] = __float2half(w1.x);
            b[2] = __float2half(-w2.y);
            b[3] = __float2half(w2.x);
            b[4] = __float2half(-w3.y);
            b[5] = __float2half(w3.x);
            b[6] = __float2half(-w4.y);
            b[7] = __float2half(w4.x);
        } else {
            b[0] = __float2half(w1.x);
            b[1] = __float2half(w1.y);
            b[2] = __float2half(w2.x);
            b[3] = __float2half(w2.y);
            b[4] = __float2half(w3.x);
            b[5] = __float2half(w3.y);
            b[6] = __float2half(w4.x);
            b[7] = __float2half(w4.y);
        }
    }
}


template <unsigned int N, unsigned int LOG2N, bool inverse>
__global__ void fft_kernel_half(half2* __restrict__ d_data, unsigned int inside_repeats) {
    // Tensor core tile: m16n16k16 (A:half, B:half, C/D:float)
    constexpr unsigned int batch            = 16;
    constexpr unsigned int warp_size        = 32;
    constexpr unsigned int ept              = (N * batch) / warp_size;   // elements per thread (if needed)
    constexpr unsigned int element_per_frag = 4;                          // per-lane fragment size for this MMA path
    constexpr unsigned int radix = 8;

    // (optional) compile-time sanity checks
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    // static_assert((1u << LOG2N) == N, "LOG2N must equal log2(N)"); // 필요하면 활성화

    (void)ept;
    (void)inside_repeats;

    // Dynamic shared memory for this block (typed). Size is provided at kernel launch.
    extern __shared__ half2 __smem[];
    auto smem = __smem + N * batch * threadIdx.y;

    // Per-lane constants/temporaries
    const int laneid = threadIdx.x & 31;

    // Zero fragment for the MMA accumulator C
    half reg_frag_zero[element_per_frag*2];
    for (int t = 0; t < element_per_frag*2; ++t) reg_frag_zero[t] = __float2half(0.0f);

    // --- Load: gmem -> smem (사용자 구현부) ---
    for(int j=0; j<batch; j++) {
        for (int i = threadIdx.x; i < N; i+=warp_size) {
            smem[i+ N * j] = d_data[reverse_3bit_groups<LOG2N>(i) + N * j +threadIdx.y * N * batch + blockIdx.x * blockDim.y * (N * batch)];
        }
    }
    __syncwarp();




    auto smem_batch0 = smem + N * (threadIdx.x / 4);
    auto smem_batch1 = smem + N * (threadIdx.x / 4) + N * (batch / 2);

    for(int iter=0; iter<inside_repeats; iter++) {
    for (unsigned int i = 0; i < LOG2N; i += 3) {
        const int stride = 1 << i;
        for(int j = 0; j < N ; j += stride * 8) {
            for(int k=0; k<stride; k++) {
                // j+k, j+k+stride, j+k+2*stride, j+k+3*stride, j+k+4*stride, j+k+5*stride, j+k+6*stride, j+k+7*stride
                // MMA input/output fragments
                half2 reg_frag_a[element_per_frag];
                half2 reg_frag_b[element_per_frag];
                half2 reg_frag_d[element_per_frag];

                reg_frag_a[0] = smem_batch0[j + k + stride * ((threadIdx.x%4))];
                reg_frag_a[2] = smem_batch0[j + k + stride * ((threadIdx.x%4)+4)];

                reg_frag_a[1] = smem_batch1[j + k + stride * ((threadIdx.x%4))];
                reg_frag_a[3] = smem_batch1[j + k + stride * ((threadIdx.x%4)+4)];

                // --- Fill per-lane fragments from smem (사용자 제공 헬퍼) ---
                fill_reg_b<false>(reg_frag_b, stride, k);

                // --- MMA: D = A x B + C ---
                mma_m16n16k16_fp16_fp16_rowcol(
                    (unsigned int*)reg_frag_d, (unsigned int*)reg_frag_a, (unsigned int*)reg_frag_b, (unsigned int*)reg_frag_zero, (void*)(smem+sizeof(half2)*N*batch));

                // --- Store result fragment back to smem (사용자 제공 헬퍼) ---

                smem_batch0[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[0];
                smem_batch0[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[2];

                smem_batch1[j + k + stride * ((threadIdx.x%4))] = reg_frag_d[1];
                smem_batch1[j + k + stride * ((threadIdx.x%4)+4)] = reg_frag_d[3];
            }
        }
        __syncwarp();
    }

    }
    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0) {
    //     for(int i=0;i<64;i++) {
    //         printf("%d: ", i);
    //         for(int j=0;j<64;j++) {
    //             printf("(%.3f,%.3f)", __half2float(smem[i*64+j].x), __half2float(smem[i*64+j].y));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    // --- Store: smem -> gmem (사용자 구현부) ---
    // TODO: 최종 결과를 d_data로 내보내기
    // ...
}

template<unsigned int N>
void my_fft_sm_half(half2 *d_data, unsigned int B) {
    static constexpr unsigned int inside_repeats = 1000;
    static constexpr unsigned int kernel_runs    = 1;
    static constexpr unsigned int warm_up_runs   = 0;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

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

    static constexpr unsigned int warp_per_block = 4;

    double elapsed_time_repeat = measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_kernel_half<N,6,false><<<B / (16 * warp_per_block), dim3(32, warp_per_block), 16*N*sizeof(half2)*warp_per_block, stream>>>(d_data, inside_repeats);
            // assert("4096 half is not supported" && false);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double elapsed_time_repeatx2 = measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_kernel_half<N,6,false><<<B / (16 * warp_per_block), 32 * warp_per_block, 16*N*sizeof(half2)*warp_per_block, stream>>>(d_data, 2 * inside_repeats);
            // assert("4096 half is not supported" && false);
        },
        warm_up_runs,
        kernel_runs,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // std::cout << "elapsed_time_repeat: " << elapsed_time_repeat << std::endl;
    printf("computation time: %.8f ms\n", (elapsed_time_repeatx2-elapsed_time_repeat)/inside_repeats);
}