#include "gpuTimer.h"
#include <cufft.h>
#include <cufftdx.hpp>
#include <typeinfo>

#define DEBUG_VAR(x) std::cout << #x << ": " << (x) << std::endl;

#define checkCuda(expr)                                                        \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

using namespace cufftdx;

static constexpr unsigned int tile_size = 64;

template <typename real_type, class fft_r2c, class fft_c2r,
          class fft_c2c_forward, class fft_c2c_inverse, class fft_single>
__global__ void conv_kernel(real_type *d_input,
                            cufftdx::complex<real_type> *d_filter,
                            real_type *d_output, int N, int f) {
    using complex_type = typename cufftdx::complex<real_type>;

    // Allocate register
    complex_type thread_data[fft_c2c_forward::storage_size];

    // Define hared memory
    // Note: must be aligned. (i.g. fft_r2c::shared_memory_size is multiple of
    // sizeof(complex)*4 = 32 -> cufftdx used aligned smem) Todo: increase tile
    // size to 256, offloading data to register
    extern __shared__ __align__(alignof(float4)) complex_type shared[];
    complex_type *shared_buffer = shared;
    __shared__ complex_type tile[tile_size * (tile_size / 2 + 1)];
    // shared + (fft_c2c_forward::shared_memory_size / sizeof(complex_type));
    // float *real_tile = reinterpret_cast<float *>(tile);
    // complex_type* tile = shared;
    // complex_type* shared_buffer = shared+tile_size*(tile_size/2+1);

    // thread idx
    const int local_thread_id = threadIdx.x;
    const int local_fft_id = threadIdx.y;

    // constant
    constexpr int fpb = fft_c2c_forward::ffts_per_block;
    constexpr int c2r_ept = fft_c2c_forward::elements_per_thread / 2;

    const int input_size = N;
    const int output_size = N - f + 1;
    const int valid_tile_size = tile_size - f + 1;

    const int input_data_bias = (blockIdx.x + blockIdx.y * N) * valid_tile_size;
    const int output_data_bias =
        (blockIdx.x + blockIdx.y * output_size) * valid_tile_size;

    // r2c
    for (int j = 0; j < (tile_size - 1) / fpb + 1; j++) {
        int local_row = local_fft_id + fpb * j;
        int global_row = local_row + blockIdx.y * valid_tile_size;

        if (local_row < tile_size) {
            auto d_input_asComplex = reinterpret_cast<complex_type *>(
                d_input + input_data_bias + local_row * input_size);
            for (int i = 0; i < (tile_size / 2 + 1 - 1) / fft_r2c::stride + 1;
                 i++) {
                int col = local_thread_id + fft_r2c::stride * i;
                if (global_row < input_size &&
                    col * 2 + blockIdx.x * valid_tile_size < input_size)
                    thread_data[i] = d_input_asComplex[col];
                else
                    thread_data[i] = complex_type(0, 0);
            }
        }
        // fft
        fft_r2c().execute(thread_data, shared_buffer);
        // // register -> smem;
        if (local_row < tile_size) {
            for (int i = 0; i < (tile_size / 2 + 1 - 1) / fft_r2c::stride + 1;
                 i++) {
                int col = local_thread_id + fft_r2c::stride * i;

                if (col < tile_size / 2 + 1)
                    tile[col + (tile_size / 2 + 1) * local_row] =
                        thread_data[i];
            }
        }
    }

    __syncthreads();
    // c2c forward & backward
    for (int j = 0; j < (tile_size / 2 + 1 - 1) / fpb + 1; j++) { // 17->16
        int col = local_fft_id + fpb * j;
        // smem -> register
        if (col < tile_size / 2 + 1) {
            for (int i = 0; i < fft_c2c_forward::elements_per_thread; i++) {
                thread_data[i] =
                    tile[(local_thread_id + fft_c2c_forward::stride * i) *
                             (tile_size / 2 + 1) +
                         col];
            }
        }

        // fft
        fft_c2c_forward().execute(thread_data, shared_buffer);

        // element wise mult
        if (col < tile_size / 2 + 1) {
            for (int i = 0; i < fft_c2c_forward::elements_per_thread; i++) {
                thread_data[i] =
                    thread_data[i] *
                    d_filter[(local_thread_id + fft_c2c_forward::stride * i) *
                                 (tile_size / 2 + 1) +
                             col];
            }
        }

        // fft
        fft_c2c_inverse().execute(thread_data, shared_buffer);

        // register -> smem
        if (col < tile_size / 2 + 1) {
            for (int i = 0; i < fft_c2c_forward::elements_per_thread; i++) {
                tile[(local_thread_id + fft_c2c_forward::stride * i) *
                         (tile_size / 2 + 1) +
                     col] = thread_data[i];
                // tile[(local_thread_id+fft_c2c_forward::stride*i)+col*tile_size]=thread_data[i];
            }
        }
    }
    __syncthreads();

    for (int j = 0; j < (valid_tile_size - 1) / fpb + 1; j++) {
        int local_row = local_fft_id + fpb * j;
        int global_row = local_row + blockIdx.y * valid_tile_size;
        // register -> smem;
        if (local_row < valid_tile_size && global_row < output_size) {
            for (int i = 0; i < (tile_size / 2 + 1 - 1) / fft_r2c::stride + 1;
                 i++) {
                int col = local_thread_id + fft_r2c::stride * i;
                if (col < tile_size / 2 + 1) {
                    thread_data[i] =
                        tile[col + (tile_size / 2 + 1) * local_row];
                    // thread_data_x[i] = tile[col*tile_size+local_row];
                }
            }
        }

        // fft
        fft_c2r().execute(thread_data, shared_buffer);

        // global mem -> register
        // if(local_row<valid_tile_size && global_row<output_size) {
        //   auto d_output_asComplex = reinterpret_cast<complex_type *>(
        //       d_output + output_data_bias + local_row * output_size);
        //   for (int i = 0, idx = local_thread_id;A idx < valid_tile_size / 2;
        //       i++, idx += fft_c2r::stride) {
        //     d_output_asComplex[idx] = thread_data[i];
        //   }
        // }

        if (local_row < valid_tile_size && global_row < output_size) {
            auto d_output_asComplex = reinterpret_cast<complex_type *>(
                d_output + output_data_bias + local_row * output_size);
            for (int i = 0; i < (tile_size / 2 + 1 - 1) / fft_r2c::stride + 1;
                 i++) {
                int local_col = local_thread_id + fft_r2c::stride * i;
                int global_col = local_col * 2 + blockIdx.x * valid_tile_size;
                if (local_col < valid_tile_size / 2 &&
                    global_col < output_size) {
                    d_output_asComplex[local_col] = thread_data[i];
                    // tile[local_col+(tile_size/2+1)*local_row] =
                    // thread_data[i];
                }
            }
        }
    }
}

template <class FFT> void print_FFT_info() {
    std::cout << "FFT::storage_size: " << FFT::storage_size << std::endl;
    std::cout << "FFT::shared_memory_size: " << FFT::shared_memory_size
              << std::endl;
    std::cout << "FFT::requires_workspace: " << FFT::requires_workspace
              << std::endl;
    std::cout << "FFT::stride: " << FFT::stride << std::endl;
    DEBUG_VAR(FFT::shared_memory_size);
    DEBUG_VAR(FFT::elements_per_thread);
    std::cout << "FFT::block_dim: (" << FFT::block_dim.x << ","
              << FFT::block_dim.y << "," << FFT::block_dim.z << ")"
              << std::endl;
}

template <typename real_type>
void FFTconv(real_type *d_input, cufftdx::complex<real_type> *d_filter,
             real_type *d_output, int N, int f) {
    using complex_type = cufftdx::complex<real_type>;
    static constexpr unsigned int Arch = CUFFT_TARGET_ARCHS;

    // Kernel settings
    static constexpr unsigned int ept = 4;  // element per thread
    static constexpr unsigned int fpb = 16; // fft per block

    using fft_base = decltype(Block() + Precision<real_type>() + SM<Arch>() +
                              Size<tile_size>() + ElementsPerThread<ept>() +
                              FFTsPerBlock<fpb>());

    using fft_r2c =
        decltype(fft_base() + Type<fft_type::r2c>() +
                 RealFFTOptions<complex_layout::natural, real_mode::folded>());
    using fft_c2r =
        decltype(fft_base() + Type<fft_type::c2r>() +
                 RealFFTOptions<complex_layout::natural, real_mode::folded>());
    using fft_c2c_forward = decltype(fft_base() + Type<fft_type::c2c>() +
                                     Direction<fft_direction::forward>());
    using fft_c2c_inverse = decltype(fft_base() + Type<fft_type::c2c>() +
                                     Direction<fft_direction::inverse>());

    using fft_single = decltype(
        Block() + Precision<real_type>() + SM<Arch>() + Size<tile_size>() +
        ElementsPerThread<ept / fpb>() + FFTsPerBlock<1>() +
        Type<fft_type::c2c>() + Direction<fft_direction::forward>());

    // std::cout << "fft_r2c" << std::endl; print_FFT_info<fft_r2c>();
    // std::cout << "fft_c2r" << std::endl; print_FFT_info<fft_c2r>();
    // std::cout << "fft_c2c_forward" << std::endl;
    // print_FFT_info<fft_c2c_forward>(); std::cout << "fft_c2c_backward" <<
    // std::endl; print_FFT_info<fft_c2c_inverse>(); std::cout << "fft_single"
    // << std::endl; print_FFT_info<fft_single>();

    constexpr size_t total_shared_mem = fft_c2c_forward::shared_memory_size;
    cudaFuncSetAttribute(
        conv_kernel<real_type, fft_r2c, fft_c2r, fft_c2c_forward,
                    fft_c2c_inverse, fft_single>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, total_shared_mem);

    int tile_num = (N + tile_size - f - f + 1) / (tile_size - f + 1);
    dim3 tile_grid(tile_num, tile_num);
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    GpuTimer timer(stream);
    timer.Start();
    // shrinkage_kernel<real_type, fft_r2c, fft_c2r, fft_c2c_forward,
    //                   fft_c2c_inverse,fft_single><<<tile_num*tile_num,
    //                   fft_r2c::block_dim, total_shared_mem>>>(
    //     d_input, d_filter, d_output, N, f);
    for (int i = 0; i < 1000; i++) {
        conv_kernel<real_type, fft_r2c, fft_c2r, fft_c2c_forward,
                    fft_c2c_inverse, fft_single>
            <<<tile_grid, fft_r2c::block_dim, total_shared_mem, stream>>>(
                d_input, d_filter, d_output, N, f);
    }
    timer.Stop();
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamDestroy(stream));

    float time_ms = timer.Elapsed();
    int stride = tile_size - f + 1;
    int tiles_per_dim = (N - f + stride) / stride;
    long long ops_per_tile = static_cast<long long>(
        4 * tile_size * tile_size * log2(tile_size * tile_size) +
        6 * tile_size * tile_size);
    long long total_ops =
        static_cast<long long>(tiles_per_dim) * tiles_per_dim * ops_per_tile;
    float gflops = total_ops / (time_ms * 1e6f);

    printf("[cuFFTDx] Time: %.3f ms, GFLOPS: %.2f\n", time_ms / 1000, gflops * 1000);

    // static float total_time=0;
    // static int cnt=0;
    // total_time+=time_ms;
    // cnt++;
    // printf("Avg Time: %.3f ms\n", total_time/cnt);
}

template <typename real_type>
cufftdx::complex<real_type> *preprocess_filter(real_type *h_filter, int f,
                                               int T) {
    // 1. 필터 패딩 및 wrap-around 중심 정렬
    float *padded_filter = (float *)calloc(T * T, sizeof(float));
    for (int i = 0; i < f; ++i)
        for (int j = 0; j < f; ++j)
            padded_filter[((T - i) % T) * T + (T - j) % T] =
                h_filter[i * f + j] / T / T;

    // 2. GPU 메모리 할당
    float *d_filter;
    cufftComplex *d_filter_fft;
    cudaMalloc(&d_filter, sizeof(float) * T * T);
    cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * T * (T / 2 + 1));

    // 3. 복사
    cudaMemcpy(d_filter, padded_filter, sizeof(float) * T * T,
               cudaMemcpyHostToDevice);
    free(padded_filter);

    // 4. FFT plan 생성 및 실행
    cufftHandle plan;
    cufftPlan2d(&plan, T, T, CUFFT_R2C);
    cufftExecR2C(plan, d_filter, d_filter_fft);
    cufftDestroy(plan);

    // 5. 중간 입력 버퍼 해제
    cudaFree(d_filter);

    // 6. 결과 반환
    return reinterpret_cast<cufftdx::complex<real_type> *>(d_filter_fft);
}

void convolution_cufftdx(float *h_input, float *h_filter, float *h_output,
                         int N, int f) {
    float *d_input, *d_output;
    int out_size = N - f + 1;

    cudaMalloc(&d_input, N * N * sizeof(float));
    cudaMalloc(&d_output, out_size * out_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto d_filter = preprocess_filter(h_filter, f, tile_size);

    FFTconv<float>(d_input, d_filter, d_output, N, f);

    // Copy result back to host
    checkCuda(cudaMemcpy(h_output, d_output,
                         out_size * out_size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Free memory
    checkCuda(cudaFree(d_input));
    checkCuda(cudaFree(d_output));
    checkCuda(cudaFree(d_filter));
}