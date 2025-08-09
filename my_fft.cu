#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                      \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define PI_F 3.14159265358979323846f

using namespace nvcuda;

__device__ __forceinline__
cuFloatComplex W(int k, int N) {
    k=(k%N+N)%N;
    float theta = - 2.0f * PI_F * k / N;
    return make_cuFloatComplex(cosf(theta), sinf(theta));
}

__device__ __forceinline__ unsigned int bit_reverse32(unsigned int x) {
    x = __brev(x); // CUDA intrinsic: 32-bit reverse
    return x;
}

__device__ __forceinline__ unsigned int bit_reverse(unsigned int x, int bits) {
    return __brev(x) >> (32 - bits);
}

__global__ void fft_kernel(cuFloatComplex* d_data, int N) {
    int tidx = threadIdx.x;

    // 공유 메모리 선언
    extern __shared__ cuFloatComplex s_data[];

    // log2(N) 계산
    int log2N = __ffs(N) - 1;

    // Global → Shared 복사
    s_data[tidx] = d_data[bit_reverse(tidx, log2N)];
    s_data[tidx+N/2] = d_data[bit_reverse(tidx+N/2, log2N)];
    __syncthreads();


    for (int stage = 0; stage < log2N; ++stage) {
        int stride = 1 << stage;
        int group_size = stride << 1;

        int group_id = tidx / stride;
        int index_in_group = tidx % stride;

        int i = group_id * group_size + index_in_group;

        cuFloatComplex a = s_data[i];
        cuFloatComplex b = s_data[i + stride];
        cuFloatComplex twiddle = W(index_in_group, group_size);

        cuFloatComplex t = cuCmulf(twiddle, b);

        __syncthreads();  // ensure a, b are all read

        s_data[i] = cuCaddf(a, t);
        s_data[i + stride] = cuCsubf(a, t);

        __syncthreads();  // update s_data before next stage
    }

    d_data[tidx] = s_data[tidx];
    d_data[tidx + N / 2] = s_data[tidx + N / 2];
}


__device__ __forceinline__ int reverse_2bit_groups(int x, int N) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

__global__ void fft_kernel_radix4(cuFloatComplex* d_data, int N) {
    int tidx = threadIdx.x;

    // 공유 메모리 선언
    extern __shared__ cuFloatComplex s_data[];

    // log2(N) 계산
    int log2N = __ffs(N) - 1;

    // Global → Shared 복사
    s_data[tidx] = d_data[reverse_2bit_groups(tidx, log2N)];
    s_data[tidx+N/4] = d_data[reverse_2bit_groups(tidx+N/4, log2N)];
    s_data[tidx+2*N/4] = d_data[reverse_2bit_groups(tidx+2*N/4, log2N)];
    s_data[tidx+3*N/4] = d_data[reverse_2bit_groups(tidx+3*N/4, log2N)];
    __syncthreads();


    for (int stage = 0; stage < log2N; stage+=2) {
        int stride = 1 << stage;
        int group_size = stride << 2;

        int group_id = tidx / stride;
        int index_in_group = tidx % stride;

        int i = group_id * group_size + index_in_group;

        cuFloatComplex a = s_data[i];
        cuFloatComplex b = s_data[i + stride];
        cuFloatComplex c = s_data[i + stride * 2];
        cuFloatComplex d = s_data[i + stride * 3];

        cuFloatComplex twiddle = W(index_in_group, group_size);

        cuFloatComplex x = cuCmulf(twiddle, b);
        cuFloatComplex y = cuCmulf(twiddle, cuCmulf(twiddle, c));
        cuFloatComplex z = cuCmulf(twiddle, cuCmulf(twiddle, cuCmulf(twiddle, d)));

        cuFloatComplex apc = cuCaddf(a,y);
        cuFloatComplex amc = cuCsubf(a,y);
        cuFloatComplex bpd = cuCaddf(x,z);
        cuFloatComplex bmd = cuCmulf(make_cuFloatComplex(0.0f, -1.0f),cuCsubf(x,z));

        s_data[i] = cuCaddf(apc, bpd);
        s_data[i + stride] = cuCaddf(amc, bmd);
        s_data[i + stride*2] = cuCsubf(apc, bpd);
        s_data[i + stride*3] = cuCsubf(amc, bmd);

        __syncthreads();  // update s_data before next stage
    }

    d_data[tidx] = s_data[tidx];
    d_data[tidx + N / 4] = s_data[tidx + N / 4];
    d_data[tidx + 2*N / 4] = s_data[tidx + 2*N / 4];
    d_data[tidx + 3*N / 4] = s_data[tidx + 3*N / 4];
}


template<int N, unsigned int warp_num=1>
__global__ void fft_kernel_radix4_matmul(cuFloatComplex* d_data) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    constexpr unsigned int log2N = 12;//__ffs(N)-1;

    // 공유 메모리 선언
    extern __shared__ cuFloatComplex s_data[];

    // Global → Shared 복사
    int total_thread=blockDim.x * blockDim.y;
    int total_idx=tidy*blockDim.x + tidx;
    
    for(int i=0; i< (N-1)/total_thread+1; i++)
        if(total_idx + total_thread * i <N)
        s_data[total_idx + total_thread * i ] = d_data[reverse_2bit_groups(total_idx + total_thread * i , log2N)];

    __syncthreads();

    // mma.sync.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> zero_frag;
    wmma::fill_fragment(zero_frag, 0.0f);
    wmma::fill_fragment(a_frag, __float2half(0.0f));
    wmma::fill_fragment(b_frag, __float2half(0.0f));
    
    __shared__ half _a_matrix[16 * 16 * warp_num];
    __shared__ half _b_matrix[16 * 16 * warp_num];
    __shared__ float _output[16 * 16 * warp_num];
    auto a_matrix = _a_matrix + 16*16*tidy;
    auto b_matrix = _b_matrix + 16*16*tidy;
    auto output = _output + 16*16*tidy;

    #pragma unroll
    for (int stage = 0; stage < log2N; stage+=2) {
        int stride = 1 << stage;
        int group_size = stride << 6;

        for(int widx = tidy; widx < N/64/*max(stride, N/64)*/; widx += warp_num) {
            int group_id = widx / stride;
            int index_in_group = widx % stride;
            int data_bias = group_id * group_size + index_in_group;
            
            //모든 쓰레드가 동일 작업 반복 중
            //어쩌피 ptx로 바꿀 예정이라
            /*
            A = i           i+stride    i+2*stride  i+3*stirde
                i+4*stride  i+5*stride  i+6*stride  i+7*stirde
                ...
                i+61*stride  i+62*stride  i+63*stride  i+64*stirde

            B^T = [ 1  w_N^k         w_N^2k           w_N^3k
                    1  w_N^(k+N/4)   w_N^(2k + 2N/4)  w_N^(3k + 3N/4)
                    1  w_N^(k+2N/4)  w_N^(2k + 4N/4)  w_N^(3k + 6N/4)
                    1  w_N^(k+3N/4)  w_N^(2k + 6N/4)  w_N^(3k + 9N/4)
                = [ 1  1  1  1    *  [ w_N^0
                    1 -i -1  i                w_N^k
                    1 -1  1 -1                        w_N^2k
                    1  i -1 -i ]                              w_N^3k]
            */
            for(int i=0; i<16; i++) {
                for(int j=0; j<4; j++) {
                    cuFloatComplex val = data_bias + stride * (4 * i + j)<N ? s_data[data_bias + stride * (4 * i + j)] : make_cuFloatComplex(0.0f,0.0f);
                    a_matrix[i * 16 + j * 2]     = __float2half(val.x);
                    a_matrix[i * 16 + j * 2 + 1] = __float2half(val.y);
                    // a_matrix[i * 16 + j * 2 + 8]  = __float2half(0.0f);
                    // a_matrix[i * 16 + j * 2 + 9]  = __float2half(0.0f);
                }
            }

            // for(int i=0; i<16; i++)
            //     for(int j=0; j<16; j++)
            //         b_matrix[i*16+j] = __float2half(0.0f);
            
            for(int i=0; i< 4; i++) {
                for(int j=0; j<4 ;j++) {
                    cuFloatComplex tw = W(i * (index_in_group + stride * j), 4 * stride);

                    b_matrix[(2 * i + 0) * 16 + j * 2 + 0] = __float2half(tw.x);
                    b_matrix[(2 * i + 0) * 16 + j * 2 + 1] = __float2half(tw.y);

                    b_matrix[(2 * i + 1) * 16 + j * 2 + 0] = __float2half(-tw.y);
                    b_matrix[(2 * i + 1) * 16 + j * 2 + 1] = __float2half(tw.x);
                }
            }
            
            wmma::load_matrix_sync(a_frag, a_matrix, 16);
            wmma::load_matrix_sync(b_frag, b_matrix, 16);

            wmma::mma_sync(c_frag, a_frag, b_frag, zero_frag);
            wmma::store_matrix_sync(output, c_frag, 16, wmma::mem_row_major);

            for(int i=0; i<16; i++) {
                for(int j=0; j<4; j++) {
                    if(data_bias + stride * (4 * i + j) < N)
                        s_data[data_bias + stride * (4 * i + j)] = make_cuFloatComplex(output[i*16+j*2], output[i*16+j*2+1]);
                }
            }
        }
        __syncthreads();
    }

    for(int i=0; i< (N-1)/total_thread+1; i++)
        if(total_idx + total_thread * i <N)
            d_data[total_idx + total_thread * i ] = s_data[total_idx + total_thread * i ];
}

static __device__ __forceinline__
void mma_m16n8k8_f32_f32_rowcol(
    float d[4],
    const float a[4],
    const float b[2],
    const float c[4],
    void *smem
){
    d[0]=a[0];
    d[1]=a[2];
    d[2]=a[1];
    d[3]=a[3];
    // asm volatile(
    //     "mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f32 "
    //     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
    //     "{%8, %9, %10, %11, %12, %13, %14, %15}, "
    //     "{%16, %17, %18, %19, %20, %21, %22, %23}, "
    //     "{%24, %25, %26, %27, %28, %29, %30, %31};\n"
    //     : // outputs (D)
    //       "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
    //       "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
    //     : // inputs A (.f16x2 -> "r"), B (.f16x2 -> "r"), C (f32 -> "f")
    //       "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
    //       "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
    //       "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
    //       "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]),
    //       "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
    //       "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7])
    // );

}

//d_data contain
__global__ void fft_kernel_radix64_batch16(cuFloatComplex* d_data) {
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

    //s
    extern __shared__ void* s;

    //Registers for data
    cuFloatComplex reg[ept];

    //Registers for mma : d = a * b + zero;
    float reg_frag_a[m*k/warp_size];
    float reg_frag_b[k*n/warp_size];
    float reg_frag_zero[m*n/warp_size];
    float reg_frag_d[m*n/warp_size];
    
    for(int i=0; i < m*n/warp_size; i++) reg_frag_zero[i]=0.0f;

    int laneid = threadIdx.x;
    for(int i=0; i<ept; i++) reg[i] = d_data[(laneid % radix) + (laneid / radix * N) + (i%(N/radix)) * radix + (i/ (N/radix)) * N * (warp_size/radix)];
    
    for(int i=0; i<iter; i++) {
        for(int j=0; j<N/radix; j++) {
            reg_frag_a[0] = reg[j].x;
            reg_frag_a[1] = reg[j+N/radix].x;
            reg_frag_a[1] = reg[j].y;
            reg_frag_a[1] = reg[j+N/radix].y;
            
            reg_frag_b[0]=0.0f;
            reg_frag_b[0]=0.0f;

            mma_m16n8k8_f32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero, s);

            reg[j].x = reg_frag_d[0];
            reg[j].y = reg_frag_d[1];
            reg[j+N/radix].x = reg_frag_d[0];
            reg[j+N/radix].y = reg_frag_d[1];
        }

        const int stride = 1<<(i<<1);//stride^iter;
        for(int j=0; j<32; j+=4) {
            auto tmp=reg[j];
            reg[j]=reg[j+1];
            reg[j+1]=reg[j+2];
            reg[j+2]=reg[j+3];
            reg[j+3]=reg[j];
        }
    }

    for(int i=0; i<ept; i++) d_data[i%(N/radix) + (i/ (N/radix)) * N * (warp_size/radix) + laneid * (N/radix)] = reg[i];
    
}

template<int N>
void my_fft(cuFloatComplex* d_data) {
    // fft_kernel<<<1, N/2, N * sizeof(cuFloatComplex)>>>(d_data, N);
    // fft_kernel_radix4<<<1, N/4, N * sizeof(cuFloatComplex)>>>(d_data, N);


    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    fft_kernel_radix64_batch16<<<1, 32, N * sizeof(cuFloatComplex)>>>(d_data);
    fft_kernel_radix64_batch16<<<1, 32, N * sizeof(cuFloatComplex)>>>(d_data);

    // 타이머 시작
    CHECK_CUDA(cudaEventRecord(start));

    constexpr unsigned int warp_num=1;
    dim3 grid(32, warp_num);
    fft_kernel_radix4_matmul<N,warp_num><<<1, grid, N * sizeof(cuFloatComplex)>>>(d_data);

    // fft_kernel_radix64_batch16<<<1, 32, N * sizeof(cuFloatComplex)>>>(d_data);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("my_fft kernel execution time: %.3f ms\n", milliseconds);

    //debug
    cuFloatComplex* tmp = (cuFloatComplex*)malloc(N * sizeof(cuFloatComplex));
    if (!tmp) {
        fprintf(stderr, "Host malloc failed\n");
        return;
    }

    // GPU → CPU 복사
    CHECK_CUDA(cudaMemcpy(tmp, d_data, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            printf("(%.1f, %.1f) ", tmp[i * 32 + j].x, tmp[i * 32 + j].y);
        }
        printf("\n");
    }
}

void a() {
    cuFloatComplex* tmp=nullptr;
    my_fft<16>(tmp);
    my_fft<1024>(tmp);
    my_fft<4096>(tmp);
}