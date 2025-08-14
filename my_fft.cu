#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "my_fft.h"

using namespace nvcuda;

__device__ __forceinline__ int reverse_2bit_groups(int x, int N) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

__device__ __forceinline__
void fill_reg_b(float b[], int stride,
                int i_perm, int j_perm, int k, const cuFloatComplex* __restrict__ W_ptr,
                bool inverse=false)
{
    // b = [ w^ (i+i_perm) ( k + N(j+j_perm)) ] ^ T

    //register mapping
    //0 4 8   ...   28
    //1 5 9
    //2 6 10
    //3 7 11  ...   31
    int i= (threadIdx.x / 8 - i_perm) & 3;
    int j= (threadIdx.x % 4 - j_perm) & 3;

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
    // cuFloatComplex w = make_cuFloatComplex(0.0f, 0.0f);
    int index = (16/stride)*((j*(k+stride*i))%(4*stride));
    auto w = W_ptr[index];

    
    if((threadIdx.x/4) & 1) {
        b[0] = w.y;
        b[1] = w.x;
    } else {
        b[0] = w.x;
        b[1] = -w.y;
    }
}

static __device__ __forceinline__
void mma_m16n8k8_tf32_f32_rowcol(
    float d[4],
    const float a[4],
    const float b[2],
    const float c[4]
){
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "        // D (also C)
        "{%4, %5, %6, %7}, "        // A (tf32 in .b32 regs)
        "{%8, %9}, "                // B (tf32 in .b32 regs)
        "{%10, %11, %12, %13};\n"       // C
        :  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        :  "r"(*reinterpret_cast<uint32_t const *>(&a[0])), "r"(*reinterpret_cast<uint32_t const *>(&a[1])), "r"(*reinterpret_cast<uint32_t const *>(&a[2])), "r"(*reinterpret_cast<uint32_t const *>(&a[3])),
           "r"(*reinterpret_cast<uint32_t const *>(&b[0])), "r"(*reinterpret_cast<uint32_t const *>(&b[1])), 
           "r"(*reinterpret_cast<uint32_t const *>(&c[0])), "r"(*reinterpret_cast<uint32_t const *>(&c[1])), "r"(*reinterpret_cast<uint32_t const *>(&c[2])), "r"(*reinterpret_cast<uint32_t const *>(&c[3]))
    );
}

template <typename T>
__device__ __forceinline__
void permute_radix4(T &a, T &b, T &c, T &d, int pattern) {
    T t0 = a, t1 = b, t2 = c, t3 = d;

    switch (pattern & 3) {
        // {0,3,2,1}
        case 0: a = t0; b = t3; c = t2; d = t1; break;
        // {1,0,3,2}
        case 1: a = t1; b = t0; c = t3; d = t2; break;
        // {2,1,0,3}
        case 2: a = t2; b = t1; c = t0; d = t3; break;
        // {3,2,1,0}
        default: a = t3; b = t2; c = t1; d = t0; break;
    }
}

//d_data contain
__global__ void fft_kernel_radix64_batch16(cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64) {
    //Tensor core shape
    constexpr int m=16;
    constexpr int n=8;
    constexpr int k=8;

    constexpr int radix = k/2; // = 4
    constexpr int iter = 3;
    constexpr int N = 64; // radix^iter
    constexpr int batch=m;
    constexpr int warp_size=32;
    constexpr int ept=N * batch / warp_size; // element_per_thread

    //Registers for data
    cuFloatComplex reg[ept];
    // cuFloatComplex reg_tmp[ept];

    //Registers for mma : d = a * b + zero;
    float reg_frag_a[m*k/warp_size];
    float reg_frag_b[k*n/warp_size];
    float reg_frag_zero[m*n/warp_size];
    float reg_frag_d[m*n/warp_size];

    __shared__ cuFloatComplex s_data[ept*(warp_size+1)];
    

    // for(int i=0; i < m*k/warp_size; i++) reg_frag_a[i]=0.0f;
    // for(int i=0; i < k*n/warp_size; i++) reg_frag_b[i]=0.0f;
    for(int i=0; i < m*n/warp_size; i++) reg_frag_zero[i]=0.0f;

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    for (int i=0; i<ept; i++) {
        s_data[i*(warp_size+1) + laneid] = d_data[block_id * N * batch + i * warp_size + laneid];
    }

    __syncwarp();
    for(int i=0; i<ept/2; i++) {
        reg[i] = s_data[(laneid/2)*(warp_size+1) + reverse_2bit_groups(i, 4)+(ept/2)*(laneid%2)];
        reg[i+ept/2] = s_data[(ept/2)*(warp_size+1) + (laneid/2)*(warp_size+1) + reverse_2bit_groups(i, 4)+(ept/2)*(laneid%2)];
    }

    
    for(int i=0; i<iter; i++) {
        const int stride = 1<<(i<<1);//4^iter;
        for(int j=0; j<N/radix; j++) {
            reg_frag_a[0] = reg[j].x;
            reg_frag_a[1] = reg[j + N/radix].x;
            reg_frag_a[2] = reg[j].y;
            reg_frag_a[3] = reg[j + N/radix].y;
            
            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if(stride>=4) j_perm=(j / (stride/4)) % radix;
            else j_perm=0;
            
            int i_perm = (j / stride) % radix;
            int k = j % stride;

            fill_reg_b(reg_frag_b, stride, i_perm, j_perm, k, W_64);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

            reg[j].x = reg_frag_d[0];
            reg[j].y = reg_frag_d[1];
            reg[j+N/radix].x = reg_frag_d[2];
            reg[j+N/radix].y = reg_frag_d[3];
        }

        if(i<iter-1){
            for(int j=0; j<32; j+=4*stride) {
                for(int k=0; k<stride; k++) {
                    // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                    // t0 t1 t2 t3
                    // 0  1  2  3       0  4  8  12
                    // 7  4  5  6       13 1  5  9
                    // 10 11 8  9	->  10 14 2  6
                    // 13 14 15 12		7  11 15 3
                    permute_radix4(reg[k+j], reg[k+j+stride], reg[k+j+stride*2], reg[k+j+stride*3], laneid & 3);
                }
            }
        }
    }

    // for(int i=0; i<ept; i++) d_data[warp_id * N * batch + i%(N/radix) + (i/ (N/radix)) * N * (warp_size/radix) + laneid * (N/radix)] = reg[i];
    
    // write to smem
    for(int i=0; i<ept/2; i++) {
        s_data[(warp_size+1)*(laneid/2) + 16*(laneid%2)+i] = reg[i];
        s_data[(ept/2)*(warp_size+1) + (warp_size+1)*(laneid/2) + 16*(laneid%2)+i] = reg[i+ept/2];
    }
    __syncwarp();

    // write to gmem
    for(int i=0; i<ept; i++) d_data[block_id * N * batch + laneid + i * warp_size] = s_data[i*(warp_size+1) + laneid];
}

// // in-place device kernel (noinline)
// __device__ __noinline__ void fft_kernel_r64_b16(cuFloatComplex* reg, const cuFloatComplex* __restrict__ W_64) {
//     auto block = cooperative_groups::this_thread_block();
//     //Tensor core shape
//     constexpr int m=16;
//     constexpr int n=8;
//     constexpr int k=8;

//     constexpr int radix = k/2; // = 4
//     constexpr int iter = 3;
//     constexpr int N = 64; // radix^iter
//     constexpr int batch=m;
//     constexpr int warp_size=32;
//     constexpr int ept=N * batch / warp_size; // element_per_thread

//     //Registers for mma : d = a * b + zero;
//     float reg_frag_a[m*k/warp_size];
//     float reg_frag_b[k*n/warp_size];
//     float reg_frag_zero[m*n/warp_size];
//     float reg_frag_d[m*n/warp_size];    

//     for(int i=0; i < m*k/warp_size; i++) reg_frag_a[i]=0.0f;
//     for(int i=0; i < k*n/warp_size; i++) reg_frag_b[i]=0.0f;
//     for(int i=0; i < m*n/warp_size; i++) reg_frag_zero[i]=0.0f;

//     int laneid = threadIdx.x;
    
//     for(int i=0; i<iter; i++) {
//         const int stride = 1<<(i<<1);//4^iter;
//         for(int j=0; j<N/radix; j++) {
//             reg_frag_a[0] = reg[j].x;
//             reg_frag_a[1] = reg[j + N/radix].x;
//             reg_frag_a[2] = reg[j].y;
//             reg_frag_a[3] = reg[j + N/radix].y;
            
//             // w = w_4stride
//             // b = [ w^ i ( k + Nj) ] ^ T
//             int j_perm;
//             if(stride>=4) j_perm=(j / (stride/4)) % radix;
//             else j_perm=0;
            
//             int i_perm = (j / stride) % radix;
//             int k = j % stride;

//             fill_reg_b(reg_frag_b, stride, i_perm, j_perm, k, W_64);
//             // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm, j_perm, k, reg_frag_b[0], reg_frag_b[1]);

//             mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

//             reg[j].x = reg_frag_d[0];
//             reg[j].y = reg_frag_d[1];
//             reg[j+N/radix].x = reg_frag_d[2];
//             reg[j+N/radix].y = reg_frag_d[3];
//         }

//         if(i<iter-1){
//             for(int j=0; j<32; j+=4*stride) {
//                 for(int k=0; k<stride; k++) {
//                     // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
//                     // t0 t1 t2 t3
//                     // 0  1  2  3       0  4  8  12
//                     // 7  4  5  6       13 1  5  9
//                     // 10 11 8  9	->  10 14 2  6
//                     // 13 14 15 12		7  11 15 3
//                     permute_radix4(reg[k+j], reg[k+j+stride], reg[k+j+stride*2], reg[k+j+stride*3], laneid & 3);
//                 }
//             }
//         }
//     }
// }

// __device__ int index_4096(int index) {

//     return ((index >> 4) << 6) | (index & 0b1111);
// }

// // blockDim = {32, 4}
// // gridDim = 1
// __global__ void fft_kernel_radix4096_batch1(cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64) {
//     constexpr int N = 4096;
//     constexpr int radix = 64;
//     constexpr int radix_unit = 64;
//     constexpr int batch_unit = 16;
//     constexpr int warp_size = 32;
//     constexpr int ept=radix * batch_unit / warp_size;
//     constexpr int num_warp = 4;

//     assert(blockDim.x == warp_size && blockDim.y == 4);

//     cuFloatComplex reg[radix_unit];

//     int warp_id = threadIdx.x / warp_size;
//     int lane_id = threadIdx.x % warp_size;

//     __shared__ cuFloatComplex s_data[num_warp*ept*(warp_size+1)];
    
//     // gmem -> smem -> reg
//     for(int i=0; i<ept; i++) {
//         s_data[warp_id*ept*(warp_size+1) + i*(warp_size+1) + lane_id] = d_data[batch_unit*warp_id + index_4096(i*warp_size + lane_id)];
//     }
//     __syncwarp();

//     for(int i=0; i<ept/2; i++) {
//         reg[i] = s_data[(lane_id/2)*(warp_size+1) + reverse_2bit_groups(i, 4)+(ept/2)*(lane_id%2)];
//         reg[i+ept/2] = s_data[(ept/2)*(warp_size+1) + (lane_id/2)*(warp_size+1) + reverse_2bit_groups(i, 4)+(ept/2)*(lane_id%2)];
//         // printf("tid: %d, i: %d, laneid: %d, reg[i]: %f %f, reg[i+ept/2]: %f %f\n", threadIdx.x, i, laneid, reg[i].x, reg[i].y, reg[i+ept/2].x, reg[i+ept/2].y);
//     }

//     // fft64_b16 iter 0 execute (4 warp executes each fft parallel)
//     assert(N/radix_unit/batch_unit == 4);
//     for(int iter=0; iter<N/radix_unit/batch_unit; iter++) {
//         fft_kernel_r64_b16(reg, W_64);
//     }

//     // reg -> smem (syncthread needed not syncwarp) bank conflict not optimal
//     for(int i=0; i<ept; i++) {
        
//     }
//     __syncthreads();

//     // smem -> reg
    

//     // fft64_b16 iter 1 execute (4 warp executes each fft parallel)
//     for(int iter=0; iter<N/radix_unit/batch_unit; iter++) {
//         fft_kernel_r64_b16(reg, W_64);
//     }

//     // reg -> smem -> gmem


// }