#include "thunderfft/detail/utils.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#include <cute/tensor.hpp>

#include <cufftdx.hpp>

#include <thunderfft/thunderfft.cuh>

using namespace cute;

#ifndef CHECK_CUDA
#define CHECK_CUDA(stmt)                                                       \
    do {                                                                       \
        cudaError_t err__ = (stmt);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#define checkCufft(expr)                                                       \
    do {                                                                       \
        cufftResult err = (expr);                                              \
        if (err != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "cuFFT Error at %s:%d: %d\n", __FILE__,       \
                         __LINE__, err);                                       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename T> __device__ void swap_vals(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T> __device__ void swap_thread_data(T *thread_data) {
    swap_vals(thread_data[1], thread_data[4]);
    swap_vals(thread_data[17], thread_data[20]);
    swap_vals(thread_data[2], thread_data[8]);
    swap_vals(thread_data[18], thread_data[24]);
    swap_vals(thread_data[3], thread_data[12]);
    swap_vals(thread_data[19], thread_data[28]);
    swap_vals(thread_data[6], thread_data[9]);
    swap_vals(thread_data[22], thread_data[25]);
    swap_vals(thread_data[7], thread_data[13]);
    swap_vals(thread_data[23], thread_data[29]);
    swap_vals(thread_data[11], thread_data[14]);
    swap_vals(thread_data[27], thread_data[30]);
}

// Kernel for element-wise complex multiplication
__global__ void complex_multiply_kernel(cufftComplex* a, const cufftComplex* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
        a[idx].x = real;
        a[idx].y = imag;
    }
}

// Kernel for scaling
__global__ void scale_kernel(cufftComplex* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// thread block size: 128
template <int f>
__launch_bounds__(128)
__global__ void convolution_kernel(cufftdx::complex<float> *d_input, 
    cufftdx::complex<float> *d_filter, cufftdx::complex<float> *d_output, int N, float* dW) {
    using complex_type = cufftdx::complex<float>;

    static_assert(f == 33, "f must be 33");

    static constexpr unsigned int tile_size = 64;
    static constexpr unsigned int batch_per_warp = 16;
    static constexpr unsigned int warp_size = 32;
    static constexpr unsigned int valid_tile_size = tile_size - f + 1;
    static constexpr unsigned int ept = tile_size * batch_per_warp / warp_size;

    // Allocate register
    extern __shared__ __align__(alignof(float4)) complex_type tile[];

    // thread idx
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const int output_size = N - f + 1;

    const int input_data_bias = (blockIdx.x + blockIdx.y * N) * valid_tile_size;
    const int output_data_bias = (blockIdx.x + blockIdx.y * output_size) * valid_tile_size;

    auto input_gmem_layout = make_layout(
        make_shape(
            make_shape(Int<tile_size>{}, blockDim.x), 
            make_shape(Int<tile_size>{}, blockDim.y)
        ), 
        make_stride(
            make_stride(N, valid_tile_size*N), 
            make_stride(Int<1>{}, Int<valid_tile_size>{})
        )
    );

    auto output_gmem_layout = make_layout(
        make_shape(
            make_shape(Int<valid_tile_size>{}, blockDim.x), 
            make_shape(Int<valid_tile_size>{}, blockDim.y)
        ),
        make_stride(
            make_stride(output_size, valid_tile_size*output_size),
            make_stride(Int<1>{}, Int<valid_tile_size>{})
        )
    );

    auto filter_gmem_layout = make_layout(
        make_shape(
            Int<tile_size>{}, Int<tile_size>{}
        ),
        LayoutRight()
    );

    auto smem_layout_x = make_layout(
        make_shape(Int<tile_size>{}, make_shape(Int<tile_size/4>{}, Int<4>{})), 
        make_stride(Int<tile_size+4>{}, make_stride(Int<1>{}, Int<tile_size/4+1>{}))
    );

    auto smem_layout_y = make_layout(
        make_shape(
            make_shape(Int<tile_size/4>{}, Int<4>{}), 
            make_shape(Int<tile_size/4>{}, Int<4>{})
        ), 
        make_stride(
            make_stride(Int<tile_size+4>{}, Int<(tile_size+4)*tile_size/4+4>{}), 
            make_stride(Int<1>{}, Int<tile_size/4+1>{})
        )
    );

    auto input_gmem_tensor = make_tensor(make_gmem_ptr(d_input), input_gmem_layout);
    auto filter_gmem_tensor = make_tensor(make_gmem_ptr(d_filter), filter_gmem_layout);

    auto output_gmem_tensor = make_tensor(make_gmem_ptr(d_output), output_gmem_layout);

    auto smem_tensor_x = make_tensor(make_smem_ptr(tile), smem_layout_x);
    auto smem_tensor_y = make_tensor(make_smem_ptr(tile), smem_layout_y);

    auto input_cta_tiler = make_shape(Int<tile_size>{}, Int<tile_size>{});
    
    auto output_cta_tiler = make_shape(Int<valid_tile_size>{}, Int<valid_tile_size>{});

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);

    auto input_tile_gmem_tensor = local_tile(input_gmem_tensor, input_cta_tiler, cta_coord);
    auto output_tile_gmem_tensor = local_tile(output_gmem_tensor, output_cta_tiler, cta_coord);

    auto reg = make_tensor<cufftdx::complex<float>>(make_shape(Int<ept>{}), LayoutRight());

    // gmem -> smem    
    int row_index, col_index, reversed_row_index, reversed_col_index;

    // auto thread_layout = make_layout(make_shape(Int<4>{}, Int<32>{}), LayoutRight());
    // auto tG = local_partition(input_tile_gmem_tensor, thread_layout, threadIdx.x);
    // auto tS = local_partition(smem_tensor_x, thread_layout, threadIdx.x);
    // copy(tG, tS);
    // cp_async_fence();
    // cp_async_wait<0>();
    for (int j = 0; j < tile_size/warp_size; j++) {
        col_index = 32*j + lane_id;
        for (int i = 0; i < batch_per_warp; i++) {
            row_index = batch_per_warp*warp_id + i;
            smem_tensor_x(row_index, col_index) = input_tile_gmem_tensor(row_index, col_index);
        }
    }
    

    __syncthreads();

    // x-dim FFT: smem -> reg
    row_index = lane_id/4 + warp_id * batch_per_warp;
    for (int i = 0; i < ept/2; i++) {
        reversed_col_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
        reg(i) = smem_tensor_x(row_index, reversed_col_index);
        reg(i + ept/2) = smem_tensor_x(row_index + batch_per_warp/2, reversed_col_index);
    }
    __syncthreads();

    // dW - d_input is trash value
    thunderfft::detail::unit::fft_kernel_r64_b16<true>(reinterpret_cast<float*>(reg.data()), dW);
    
    // x-dim FFT: reg -> smem
    for (int i = 0; i < ept/2; i++) {
        int col_index = i + (ept/2)*(lane_id % 4);
        smem_tensor_y(row_index, col_index) = reg(i);
        smem_tensor_y(row_index + batch_per_warp/2, col_index) = reg(i + ept/2);
    }

    __syncthreads();

    // y-dim FFT: smem -> reg
    col_index = lane_id/4 + warp_id * batch_per_warp;
    for (int i = 0; i < ept/2; i++) {
        reversed_row_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
        reg(i) = smem_tensor_y(reversed_row_index, col_index);
        reg(i + ept/2) = smem_tensor_y(reversed_row_index, col_index + batch_per_warp/2);
    }

    __syncthreads();

    thunderfft::detail::unit::fft_kernel_r64_b16<true>(reinterpret_cast<float*>(reg.data()), dW);

    for (int i = 0; i < ept/2; i++) {
        row_index = i + (ept/2)*(lane_id % 4);
        reg(i) *= filter_gmem_tensor(row_index, col_index);
        reg(i + ept/2) *= filter_gmem_tensor(row_index, col_index + batch_per_warp/2);
    }

    swap_thread_data(reg.data());

    // // y-dim FFT: reg -> smem
    // for (int i = 0; i < ept/2; i++) {
    //     int row_index = i + (ept/2)*(lane_id % 4);
    //     warp_smem_tensor_y(row_index, lane_id/4) = reg(i);
    //     warp_smem_tensor_y(row_index, lane_id/4 + batch_per_warp/2) = reg(i + ept/2);
    // }

    // __syncthreads();
    
    // // complex element-wise multiplication
    // for (int j = 0; j < tile_size/warp_size; j++) {
    //     int col_index = 32*j + lane_id;
    //     for (int i = 0; i < batch_per_warp; i++) {
    //         int row_index = batch_per_warp*warp_id + i;
    //         smem_tensor(row_index, col_index) *= filter_gmem_tensor(row_index, col_index);
    //     }
    // }

    // __syncthreads();

    // // y-dim IFFT: smem -> reg
    // for (int i = 0; i < ept/2; i++) {
    //     reversed_row_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
    //     reg(i) = warp_smem_tensor_y(reversed_row_index, lane_id/4);
    //     reg(i + ept/2) = warp_smem_tensor_y(reversed_row_index, lane_id/4 + batch_per_warp/2);
    // }

    // __syncthreads();

    thunderfft::detail::unit::fft_kernel_r64_b16<false>(reinterpret_cast<float*>(reg.data()), dW);

    // y-dim IFFT: reg -> smem
    for (int i = 0; i < ept/2; i++) {
        row_index = i + (ept/2)*(lane_id % 4);
        smem_tensor_y(row_index, col_index) = reg(i);
        smem_tensor_y(row_index, col_index + batch_per_warp/2) = reg(i + ept/2);
    }

    __syncthreads();

    if (warp_id < 2) {
        // x-dim IFFT: smem -> reg
        row_index = lane_id/4 + warp_id * batch_per_warp;
        for (int i = 0; i < ept/2; i++) {
            reversed_col_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
            reg(i) = smem_tensor_y(row_index, reversed_col_index);
            reg(i + ept/2) = smem_tensor_y(row_index + batch_per_warp/2, reversed_col_index);
        }

        __syncthreads();

        thunderfft::detail::unit::fft_kernel_r64_b16<false>(reinterpret_cast<float*>(reg.data()), dW);
        
        // x-dim IFFT: reg -> smem
        for (int i = 0; i < ept/2; i++) {
            int col_index = i + (ept/2)*(lane_id % 4);
            smem_tensor_x(row_index, col_index) = reg(i);
            smem_tensor_x(row_index + batch_per_warp/2, col_index) = reg(i + ept/2);
        }
    }

    __syncthreads();

    // smem -> gmem
    if (warp_id < 2) {
        int col_index = lane_id;
        for (int i = 0; i < batch_per_warp; i++) {
            row_index = i + warp_id * batch_per_warp;
            output_tile_gmem_tensor(row_index, col_index) = smem_tensor_x(row_index, col_index);
        }
    }
}

static void my_convolution(const float2* h_input, const float2* h_filter, 
                           float2* h_output, int N, int f) {
    cufftdx::complex<float>* d_input = nullptr;
    cufftdx::complex<float>* d_filter = nullptr;
    cufftdx::complex<float>* d_output = nullptr;
    static constexpr unsigned int tile_size = 64;
    const unsigned int valid_tile_size = tile_size - f + 1;
    int out_size = N - f + 1;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cufftdx::complex<float>) * N * N));
    // Filter needs to be padded to tile_size x tile_size for FFT-based convolution
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(cufftdx::complex<float>) * tile_size * tile_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(cufftdx::complex<float>) * out_size * out_size));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float2) * N * N, cudaMemcpyHostToDevice));
    
    // Pad filter to tile_size x tile_size (zero-padded)
    cufftdx::complex<float>* h_filter_padded = (cufftdx::complex<float>*)std::calloc(tile_size * tile_size, sizeof(cufftdx::complex<float>));
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            // Flip filter and place at wrap-around position for circular convolution
            int idx_i = (tile_size - i) % tile_size;
            int idx_j = (tile_size - j) % tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].x = h_filter[i * f + j].x / tile_size / tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].y = h_filter[i * f + j].y / tile_size / tile_size;
        }
    }
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter_padded, sizeof(cufftdx::complex<float>) * tile_size * tile_size, cudaMemcpyHostToDevice));
    std::free(h_filter_padded);

    // FFT filter
    cufftHandle plan_forward;
    checkCufft(cufftPlan2d(&plan_forward, tile_size, tile_size, CUFFT_C2C));
    checkCufft(cufftExecC2C(plan_forward, reinterpret_cast<cufftComplex*>(d_filter), reinterpret_cast<cufftComplex*>(d_filter), CUFFT_FORWARD));

    dim3 blockSize(128);
    dim3 gridSize((out_size + valid_tile_size - 1) / valid_tile_size,
                  (out_size + valid_tile_size - 1) / valid_tile_size);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    auto smem_layout_y = make_layout(
        make_shape(
            make_shape(Int<tile_size/4>{}, Int<4>{}), 
            make_shape(Int<tile_size/4>{}, Int<4>{})
        ), 
        make_stride(
            make_stride(Int<tile_size+4>{}, Int<(tile_size+4)*tile_size/4+4>{}), 
            make_stride(Int<1>{}, Int<tile_size/4+1>{})
        )
    );
    float* dW = thunderfft::preprocess_W<float>(64);
    int shared_memory_size = sizeof(cufftdx::complex<float>) * cosize_v<decltype(smem_layout_y)>;
    for (int i = 0; i < 100; i++) {
        convolution_kernel<33><<<gridSize, blockSize, shared_memory_size>>>(d_input, d_filter, d_output, N, dW);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Time: " << ms / 100.f << " ms" << std::endl;
    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeof(cufftdx::complex<float>) * out_size * out_size,
                         cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    checkCufft(cufftDestroy(plan_forward));
}

// ------------------------------
// Reference convolution using cuFFT
// ------------------------------
static void reference_convolution_cufft(const float2* h_input, const float2* h_filter, 
                                        float2* h_output, int N, int f) {
    int out_size = N - f + 1;
    
    // For valid convolution, we need to pad to size N for circular convolution
    // or use a larger size for linear convolution
    // Using N x N for simplicity (circular convolution)
    int fft_size = N;
    
    // Allocate device memory
    cufftComplex* d_input = nullptr;
    cufftComplex* d_filter = nullptr;
    cufftComplex* d_output = nullptr;
    cufftComplex* d_input_fft = nullptr;
    cufftComplex* d_filter_fft = nullptr;
    cufftComplex* d_result_fft = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_input_fft, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_result_fft, sizeof(cufftComplex) * fft_size * fft_size));
    
    // Copy input directly to device (no padding needed since N == fft_size)
    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(cufftComplex) * N * N, 
                         cudaMemcpyHostToDevice));
    
    // Zero-pad filter (f < N, so we need to pad it to N x N)
    cufftComplex* h_filter_padded = (cufftComplex*)std::calloc(fft_size * fft_size, sizeof(cufftComplex));
    
    // Copy filter to padded array with wrap-around (circular convolution)
    // For valid convolution, we need to flip and pad the filter
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            // Flip filter and place at wrap-around position for circular convolution
            int idx_i = (fft_size - i) % fft_size;
            int idx_j = (fft_size - j) % fft_size;
            h_filter_padded[idx_i * fft_size + idx_j].x = h_filter[i * f + j].x;
            h_filter_padded[idx_i * fft_size + idx_j].y = h_filter[i * f + j].y;
        }
    }
    
    // Copy padded filter to device
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter_padded, sizeof(cufftComplex) * fft_size * fft_size, 
                         cudaMemcpyHostToDevice));
    
    std::free(h_filter_padded);
    
    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;
    checkCufft(cufftPlan2d(&plan_forward, fft_size, fft_size, CUFFT_C2C));
    checkCufft(cufftPlan2d(&plan_inverse, fft_size, fft_size, CUFFT_C2C));
    
    // Forward FFT
    checkCufft(cufftExecC2C(plan_forward, d_input, d_input_fft, CUFFT_FORWARD));
    checkCufft(cufftExecC2C(plan_forward, d_filter, d_filter_fft, CUFFT_FORWARD));
    
    // Element-wise multiply
    int num_threads = 256;
    int num_blocks = (fft_size * fft_size + num_threads - 1) / num_threads;
    
    // Copy input_fft to result_fft for multiplication
    CHECK_CUDA(cudaMemcpy(d_result_fft, d_input_fft, sizeof(cufftComplex) * fft_size * fft_size, 
                         cudaMemcpyDeviceToDevice));
    
    // Launch multiplication kernel
    complex_multiply_kernel<<<num_blocks, num_threads>>>(d_result_fft, d_filter_fft, fft_size * fft_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Inverse FFT
    checkCufft(cufftExecC2C(plan_inverse, d_result_fft, d_output, CUFFT_INVERSE));
    
    // Scale by 1/(N*N) for inverse FFT normalization
    float scale = 1.0f / (fft_size * fft_size);
    scale_kernel<<<num_blocks, num_threads>>>(d_output, scale, fft_size * fft_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back
    cufftComplex* h_result = (cufftComplex*)std::malloc(sizeof(cufftComplex) * fft_size * fft_size);
    CHECK_CUDA(cudaMemcpy(h_result, d_output, sizeof(cufftComplex) * fft_size * fft_size, 
                         cudaMemcpyDeviceToHost));
    
    // Extract valid region (top-left out_size x out_size)
    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < out_size; ++j) {
            h_output[i * out_size + j].x = h_result[i * fft_size + j].x;
            h_output[i * out_size + j].y = h_result[i * fft_size + j].y;
        }
    }
    
    // Cleanup
    std::free(h_result);
    checkCufft(cufftDestroy(plan_forward));
    checkCufft(cufftDestroy(plan_inverse));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_input_fft));
    CHECK_CUDA(cudaFree(d_filter_fft));
    CHECK_CUDA(cudaFree(d_result_fft));
}

// ------------------------------
// Generate test input (complex, using float2)
// ------------------------------
static void make_test_input(float2* h_input, int N) {
    // Initialize as row * 1000000 + col for real part, row * 1000 + col for imaginary part
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_input[i * N + j].x = float(i+j);
            h_input[i * N + j].y = float(2*i-j);
        }
    }
}

// ------------------------------
// Generate test filter (complex, using float2)
// ------------------------------
static void make_test_filter(float2* h_filter, int f) {
    // Simple pattern: Gaussian-like filter
    float center = float(f) / 2.0f;
    float sigma = float(f) / 4.0f;
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            float di = float(i) - center;
            float dj = float(j) - center;
            float val_real = std::exp(-(di*di + dj*dj) / (2.0f * sigma * sigma));
            float val_imag = val_real * 0.1f; // Small imaginary component
            h_filter[i * f + j].x = val_real;
            h_filter[i * f + j].y = val_imag;
            sum_real += val_real;
            sum_imag += val_imag;
        }
    }
    
    // Normalize
    for (int i = 0; i < f * f; ++i) {
        h_filter[i].x /= sum_real;
        h_filter[i].y /= sum_imag;
    }
}

// ------------------------------
// Validation: L2 relative error (complex)
// ------------------------------
static double l2_rel(const float2* a, const float2* b, size_t n) {
    long double num = 0.0L, den = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double diff_real = (long double)a[i].x - (long double)b[i].x;
        const long double diff_imag = (long double)a[i].y - (long double)b[i].y;
        const long double diff_mag = diff_real * diff_real + diff_imag * diff_imag;
        num += diff_mag;
        const long double ref_mag = (long double)b[i].x * (long double)b[i].x + 
                                     (long double)b[i].y * (long double)b[i].y;
        den += ref_mag;
    }
    return den > 0 ? (double)std::sqrt((double)(num/den)) : (double)std::sqrt((double)num);
}

// ------------------------------
// Validation: Linf relative error (complex)
// ------------------------------
static double linf_rel(const float2* a, const float2* b, size_t n) {
    long double max_diff = 0.0L, max_ref = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double diff_real = (long double)a[i].x - (long double)b[i].x;
        const long double diff_imag = (long double)a[i].y - (long double)b[i].y;
        const long double diff_mag = std::sqrt(diff_real * diff_real + diff_imag * diff_imag);
        const long double ref_mag = std::sqrt((long double)b[i].x * (long double)b[i].x + 
                                               (long double)b[i].y * (long double)b[i].y);
        if (diff_mag > max_diff) max_diff = diff_mag;
        if (ref_mag > max_ref)  max_ref  = ref_mag;
    }
    return max_ref > 0 ? (double)(max_diff / max_ref) : (double)max_diff;
}

// ------------------------------
// Compute checksum (complex)
// ------------------------------
static double checksum(const float2* v, size_t n) {
    long double acc_real = 0.0L, acc_imag = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        acc_real += (long double)v[i].x * (1.0L + (long double)i * 1e-9L);
        acc_imag += (long double)v[i].y * (1.0L + (long double)i * 1e-9L);
    }
    return (double)(acc_real + acc_imag * 1e-6L);
}

int main(int argc, char** argv) {
    // Configuration
    int N = 16384;      // Input image size: N x N
    int f = 33;        // Filter size: f x f
    int device = 0;
    int warmup_iters = 0;
    int timed_iters = 1;
    
    // Parse command line arguments
    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) f = std::atoi(argv[2]);
    if (argc >= 4) device = std::atoi(argv[3]);
    
    int out_size = N - f + 1;
    
    std::cout << "Convolution Benchmark: Custom vs cuFFT Reference\n"
              << "  Input size : " << N << " x " << N << "\n"
              << "  Filter size: " << f << " x " << f << "\n"
              << "  Output size: " << out_size << " x " << out_size << "\n"
              << "  Device     : " << device << "\n";

    CHECK_CUDA(cudaSetDevice(device));
    
    // Allocate host memory (complex using float2)
    const size_t input_bytes = sizeof(float2) * N * N;
    const size_t filter_bytes = sizeof(float2) * f * f;
    const size_t output_bytes = sizeof(float2) * out_size * out_size;
    
    float2* h_input = static_cast<float2*>(std::malloc(input_bytes));
    float2* h_filter = static_cast<float2*>(std::malloc(filter_bytes));
    float2* h_output_my = static_cast<float2*>(std::malloc(output_bytes));
    float2* h_output_ref = static_cast<float2*>(std::malloc(output_bytes));
    
    if (!h_input || !h_filter || !h_output_my || !h_output_ref) {
        std::fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    
    // Generate test data
    make_test_input(h_input, N);
    make_test_filter(h_filter, f);
    
    // ====================================================================
    // Run your custom convolution (with timing)
    // ====================================================================
    std::cout << "\n--- Running Custom Convolution ---\n";
    
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        my_convolution(h_input, h_filter, h_output_my, N, f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start_my, stop_my;
    CHECK_CUDA(cudaEventCreate(&start_my));
    CHECK_CUDA(cudaEventCreate(&stop_my));
    CHECK_CUDA(cudaEventRecord(start_my));
    for (int i = 0; i < timed_iters; ++i) {
        my_convolution(h_input, h_filter, h_output_my, N, f);
    }
    CHECK_CUDA(cudaEventRecord(stop_my));
    CHECK_CUDA(cudaEventSynchronize(stop_my));
    float ms_my = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_my, start_my, stop_my));
    ms_my /= float(timed_iters);
    
    // Compute GFLOPS for custom convolution
    long long ops_my = 2LL * out_size * out_size * f * f;
    float gflops_my = ops_my / (ms_my * 1e6f);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[Custom] Time: " << ms_my << " ms, GFLOPS: " 
              << std::setprecision(2) << gflops_my << std::setprecision(6) << "\n";
    
    // ====================================================================
    // Run cuFFT reference convolution (with timing)
    // ====================================================================
    std::cout << "\n--- Running cuFFT Reference Convolution ---\n";
    
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        reference_convolution_cufft(h_input, h_filter, h_output_ref, N, f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start_ref{}, stop_ref{};
    CHECK_CUDA(cudaEventCreate(&start_ref));
    CHECK_CUDA(cudaEventCreate(&stop_ref));
    CHECK_CUDA(cudaEventRecord(start_ref));
    for (int i = 0; i < timed_iters; ++i) {
        reference_convolution_cufft(h_input, h_filter, h_output_ref, N, f);
    }
    CHECK_CUDA(cudaEventRecord(stop_ref));
    CHECK_CUDA(cudaEventSynchronize(stop_ref));
    float ms_ref = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_ref, start_ref, stop_ref));
    ms_ref /= float(timed_iters);
    
    // Compute GFLOPS for cuFFT reference
    long long ops_ref = 2LL * out_size * out_size * f * f;
    float gflops_ref = ops_ref / (ms_ref * 1e6f);
    
    std::cout << "[cuFFT Reference] Time: " << ms_ref << " ms, GFLOPS: " 
              << std::setprecision(2) << gflops_ref << std::setprecision(6) << "\n";
    
    // ====================================================================
    // Validate results
    // ====================================================================
    std::cout << "\n--- Validation ---\n";
    
    const size_t output_count = out_size * out_size;
    const double err_l2 = l2_rel(h_output_my, h_output_ref, output_count);
    const double err_linf = linf_rel(h_output_my, h_output_ref, output_count);
    
    std::cout << "Checksum(Custom): " << checksum(h_output_my, output_count) << "\n";
    std::cout << "Checksum(cuFFT) : " << checksum(h_output_ref, output_count) << "\n";
    std::cout << "L2 relative error   : " << std::setprecision(6) << err_l2 << "\n";
    std::cout << "Linf relative error : " << std::setprecision(6) << err_linf << "\n";
    
    // Show first few samples
    const int to_print = std::min(64, out_size);
    std::cout << "\nFirst " << to_print << " x " << to_print << " samples:\n";
    std::cout << "  i  j     Custom(real)  Custom(imag)  cuFFT(real)   cuFFT(imag)   Diff\n";
    for (int i = 0; i < to_print; ++i) {
        for (int j = 0; j < to_print; ++j) {
            int idx = i * out_size + j;
            float my_real = h_output_my[idx].x;
            float my_imag = h_output_my[idx].y;
            float ref_real = h_output_ref[idx].x;
            float ref_imag = h_output_ref[idx].y;
            float diff = std::sqrt((my_real - ref_real) * (my_real - ref_real) + 
                                   (my_imag - ref_imag) * (my_imag - ref_imag));
            std::cout << std::setw(3) << i << " " << std::setw(3) << j 
                      << " " << std::setw(12) << my_real 
                      << " " << std::setw(12) << my_imag
                      << " " << std::setw(12) << ref_real
                      << " " << std::setw(12) << ref_imag
                      << " " << std::setw(12) << diff << "\n";
        }
    }
    
    // ====================================================================
    // Performance comparison
    // ====================================================================
    std::cout << "\n--- Performance Comparison ---\n";
    std::cout << "Speedup: " << std::setprecision(2) << (ms_ref / ms_my) << "x ";
    if (ms_my < ms_ref) {
        std::cout << "(Custom is faster)\n";
    } else {
        std::cout << "(cuFFT Reference is faster)\n";
    }
    std::cout << "Efficiency: " << std::setprecision(1) 
              << (gflops_my / gflops_ref * 100.0f) << "%\n";
    
    // ====================================================================
    // Cleanup
    // ====================================================================
    CHECK_CUDA(cudaEventDestroy(start_ref));
    CHECK_CUDA(cudaEventDestroy(stop_ref));
    CHECK_CUDA(cudaEventDestroy(start_my));
    CHECK_CUDA(cudaEventDestroy(stop_my));

    std::free(h_input);
    std::free(h_filter);
    std::free(h_output_my);
    std::free(h_output_ref);
    
    std::cout << "\nDone.\n";
    return 0;
}