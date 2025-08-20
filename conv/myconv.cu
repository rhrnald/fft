#include "gpuTimer.h"
#include "my_fft.h"

#include <cufft.h>
#include <cufftdx.hpp>
#include <typeinfo>

#define DEBUG_VAR(x) std::cout << #x << ": " << (x) << std::endl;

#define checkCuda(expr)                                                        \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

using namespace cufftdx;

static constexpr unsigned int tile_size = 64;

__global__ void my_conv_kernel(float *d_input,
                            cufftdx::complex<float> *d_filter,
                            float *d_output, int input_size, int f, const cuFloatComplex *W_4096, const cuFloatComplex *IW_4096) {
  using complex_type = typename cufftdx::complex<float>;

  // Allocate register
  float2 thread_data[32];
  __shared__ __align__(alignof(float4)) float2 tile[tile_size*(tile_size/2+1)];

  // thread idx
  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;

  const int output_size = input_size - f + 1;
  const int valid_tile_size = tile_size - f + 1;

  const int input_data_bias = (blockIdx.x + blockIdx.y * input_size) * valid_tile_size;
  const int output_data_bias =
      (blockIdx.x + blockIdx.y * output_size) * valid_tile_size;

  // gmem -> smem
  for (int local_row = warp_id; local_row < tile_size; local_row+=blockDim.y) {
    int global_row = local_row + blockIdx.y * valid_tile_size;

    int local_col = lane_id;
    int global_col = local_col*2+blockIdx.x * valid_tile_size;

    if(global_row < input_size && global_col < input_size) {
      tile[local_col + (tile_size / 2 + 1) * local_row] = *reinterpret_cast<float2 *>(d_input+global_row * input_size + global_col);
    } else {
      tile[local_col + (tile_size / 2 + 1) * local_row] = make_float2(0.0f, 0.0f);
    }
  }


  __syncthreads();

  // row-wise fft
  for(int i=0; i<32; i++) {
    int row = lane_id/4 + (i/16)*8 + warp_id*16;
    int col = lane_id%4 + (i%16)*4;
    thread_data[i].x = *(((float*)tile) +row*(tile_size + 2)+reverse_2bit_groups(col,6));
    thread_data[i].y = 0.0f;
  }

  fft_kernel_r64_b16((cuFloatComplex *)thread_data, W_4096);

  for(int i=0; i<32; i++) {
    int row = (lane_id/4) + (i/16)*8 + warp_id*16;
    int col = (lane_id%4) * 16 + (i%16);
    if(col<tile_size/2+1) {
      tile[row * (tile_size/2+1) + col] = thread_data[i];
    }
  }
  __syncthreads();

  // col-wise fft
  for(int i=0; i<32; i++) {
    int col = lane_id/4 + (i/16)*8 + warp_id*16;
    int row = lane_id%4 + (i%16)*4;
    if(col<tile_size/2+1) {
      thread_data[i] = tile[reverse_2bit_groups(row,6) * (tile_size/2+1) + col];
    }
  }

  if(warp_id<3) fft_kernel_r64_b16((cuFloatComplex *)thread_data, W_4096);

  for(int i=0; i<32; i++) {
    int col = (lane_id/4) + (i/16)*8 + warp_id*16;
    int row = (lane_id%4) * 16 + (i%16);
    if(col<tile_size/2+1) {
      tile[row * (tile_size/2+1) + col] = thread_data[i];
    }
  }
  __syncthreads();

  //element-wise mult
  for(int row=lane_id; row<tile_size; row+=32) {
    for (int col = warp_id; col <=tile_size/2; col+=blockDim.y) {
      ((complex_type*)tile)[row * (tile_size/2+1)  + col] *= d_filter[row * (tile_size/2+1)  + col];
    }
  }

  __syncthreads();
  // col-wise ifft
  for(int i=0; i<32; i++) {
    int col = lane_id/4 + (i/16)*8 + warp_id*16;
    int row = lane_id%4 + (i%16)*4;
    if(col<tile_size/2+1) {
      thread_data[i] = tile[reverse_2bit_groups(row,6) * (tile_size/2+1) + col];
    }
  }

  if(warp_id<3) fft_kernel_r64_b16((cuFloatComplex *)thread_data, IW_4096);

  for(int i=0; i<32; i++) {
    int col = (lane_id/4) + (i/16)*8 + warp_id*16;
    int row = (lane_id%4) * 16 + (i%16);
    if(col<tile_size/2+1) {
      tile[row * (tile_size/2+1) + col] = thread_data[i];
    }
  }

  __syncthreads();
  // if(local_thread_id==0 && local_fft_id==0) {
  //   for(int i=0 ; i<tile_size;i++) {
  //     for (int j = 0; j <=tile_size/2; j++) {
  //       printf("%f %f ", tile[i*(tile_size/2+1)+j].x,
  //       tile[i*(tile_size/2+1)+j].y);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n\n");
  // }
  // row-wise ifft
  for(int i=0; i<32; i++) {
    int row = lane_id/4 + (i/16)*8 + warp_id*16;
    int col = lane_id%4 + (i%16)*4;
    if(reverse_2bit_groups(col,6)<tile_size/2+1) {
      thread_data[i] = tile[row * (tile_size/2+1) + reverse_2bit_groups(col,6)];
    } else {
      thread_data[i] = tile[row * (tile_size/2+1) + tile_size-reverse_2bit_groups(col,6)];
      thread_data[i].y = -thread_data[i].y;
    }
  }

  fft_kernel_r64_b16((cuFloatComplex *)thread_data, IW_4096);

  for(int i=0; i<32; i++) {
    int row = (lane_id/4) + (i/16)*8 + warp_id*16;
    int col = (lane_id%4) * 16 + (i%16);
    *(((float*)tile) +row*(tile_size + 2)+col) = thread_data[i].x;
  }

  for (int local_row = warp_id; local_row < tile_size; local_row+=blockDim.y) {
    int global_row = local_row + blockIdx.y * valid_tile_size;

    int local_col = lane_id;
    int global_col = local_col*2+blockIdx.x * valid_tile_size;

    if(global_row < output_size && global_col < output_size && local_row < valid_tile_size && local_col*2 < valid_tile_size) {
      *reinterpret_cast<float2 *>(d_output+global_row * output_size + global_col) = tile[local_col + (tile_size / 2 + 1) * local_row];
    }
  }

  // if(local_thread_id==0 && local_fft_id==0) {
  //   for(int i=0 ; i<tile_size;i++) {
  //     for (int j = 0; j <=tile_size/2; j++) {
  //       printf("%f %f ", tile[i*(tile_size/2+1)+j].x,
  //       tile[i*(tile_size/2+1)+j].y);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n\n");
  // }
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
            << FFT::block_dim.y << "," << FFT::block_dim.z << ")" << std::endl;
}

template <typename real_type>
void my_FFTconv(real_type *d_input, cufftdx::complex<real_type> *d_filter,
                real_type *d_output, int N, int f) {
  using complex_type = cufftdx::complex<real_type>;
  // constexpr size_t total_shared_mem =
  //     sizeof(complex_type) * tile_size * (tile_size / 2 + 1);
  // cudaFuncSetAttribute(my_conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      total_shared_mem);

  cuFloatComplex h_W_4096[4096], h_IW_4096[4096];
  for (int i = 0; i < 4096; i++) {
    h_W_4096[i] = make_cuFloatComplex(cos((-2 * M_PI * i) / 4096.0),
                                      sin((-2 * M_PI * i) / 4096.0));
    
    h_IW_4096[i] = make_cuFloatComplex(cos((-2 * M_PI * i) / 4096.0),
                                      -sin((-2 * M_PI * i) / 4096.0));
  }

  cuFloatComplex *W_4096, *IW_4096;
  CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&IW_4096, 4096 * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMemcpy(IW_4096, h_IW_4096, 4096 * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));
  
  int tile_num = (N + tile_size - f - f + 1) / (tile_size - f + 1);
  dim3 tile_grid(tile_num, tile_num);
  GpuTimer timer;
  timer.Start();
  dim3 block_dim(32,4);
  my_conv_kernel<<<tile_grid, block_dim>>>(d_input, d_filter, d_output,
                                                    N, f, W_4096, IW_4096);
  timer.Stop();

  // dim3 tmp={8,1};
  // shrinkage_kernel<real_type, fft_r2c, fft_c2r, fft_c2c_forward,
  //                  fft_c2c_inverse><<<1, tmp, total_shared_mem>>>(
  //     d_input, d_filter, d_output, N, f);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaGetLastError());

  float time_ms = timer.Elapsed();
  int stride = tile_size - f + 1;
  int tiles_per_dim = (N - f + stride) / stride;
  long long ops_per_tile = static_cast<long long>(
      4 * tile_size * tile_size * log2(tile_size * tile_size) +
      6 * tile_size * tile_size);
  long long total_ops =
      static_cast<long long>(tiles_per_dim) * tiles_per_dim * ops_per_tile;
  float gflops = total_ops / (time_ms * 1e6f);

  printf("[myConv] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

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

void my_convolution(float *h_input, float *h_filter, float *h_output, int N,
                    int f) {
  float *d_input, *d_output;
  int out_size = N - f + 1;

  cudaMalloc(&d_input, N * N * sizeof(float));
  cudaMalloc(&d_output, out_size * out_size * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);
  auto d_filter = preprocess_filter(h_filter, f, tile_size);

  my_FFTconv<float>(d_input, d_filter, d_output, N, f);

  // Copy result back to host
  checkCuda(cudaMemcpy(h_output, d_output, out_size * out_size * sizeof(float),
                       cudaMemcpyDeviceToHost));

  // Free memory
  checkCuda(cudaFree(d_input));
  checkCuda(cudaFree(d_output));
  checkCuda(cudaFree(d_filter));
}