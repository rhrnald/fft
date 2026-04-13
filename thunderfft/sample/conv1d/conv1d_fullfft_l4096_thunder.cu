#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <thunderfft/thunderfft.cuh>
#include <thunderfft/thunderfft_layout_tuned.h>

#include "../utils.h"

using cpx_h = half2;

namespace {

constexpr int kN = 4096;

template <bool forward>
__global__ void thunder_fft4096_block_kernel(const cpx_h* __restrict__ d_input,
                                             cpx_h* __restrict__ d_output) {
    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    constexpr int ept = kN * BPB / (thunderfft::threads_per_warp * WPB);

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, kN, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, kN, BPB>::L_out;

    cpx_h W[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kN, forward>(W);

    thunderfft::ThunderFFT_gmem2smem<half, L_in>(s_in, d_input + blockIdx.x * BPB * kN);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, forward>(reg, W, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    thunderfft::ThunderFFT_smem2gmem<half, L_out>(d_output + blockIdx.x * BPB * kN, s_in);
}


__device__ void swap(cpx_h& a, cpx_h& b) {
    cpx_h temp = a;
    a = b;
    b = temp;
}
__device__ void reverse_data(cpx_h* data) {
    swap(data[1], data[4]);
    swap(data[2], data[8]);
    swap(data[3], data[12]);
    swap(data[6], data[9]);
    swap(data[7], data[13]);
    swap(data[11], data[14]);

    swap(data[17], data[20]);
    swap(data[18], data[24]);
    swap(data[19], data[28]);
    swap(data[22], data[25]);
    swap(data[23], data[29]);
    swap(data[27], data[30]);
}
__device__ __forceinline__ half2 cmul_h2(half2 a, half2 b) {
    return __hcmadd(a, b, __float2half2_rn(0.f));
}

template <typename sL>
__device__ __forceinline__ void ThunderFFT_gmem2smem_realio(cpx_h* __restrict__ smem,
                                                            const half* __restrict__ gmem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride = sL::batch_stride;
    constexpr int pad_period = sL::pad_period;
    constexpr int pad = sL::pad;
    static_assert(!sL::reversed, "realio loader expects non-reversed layout");

    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const half z = __float2half(0.0f);
    for (int i = 0; i < batch; ++i) {
        for (int _j = 0; _j < N; _j += thunderfft::threads_per_warp * thunderfft::warp_per_block<N>) {
            const int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx / pad_period * pad;
            smem[smem_idx] = __halves2half2(gmem[i * N + j], z);
        }
    }
}

template <typename sL>
__device__ __forceinline__ void ThunderFFT_smem2gmem_realio(half* __restrict__ gmem,
                                                            const cpx_h* __restrict__ smem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride = sL::batch_stride;
    constexpr int pad_period = sL::pad_period;
    constexpr int pad = sL::pad;
    static_assert(!sL::reversed, "realio storer expects non-reversed layout");

    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = 0; i < batch; ++i) {
        for (int _j = 0; _j < N; _j += thunderfft::threads_per_warp * thunderfft::warp_per_block<N>) {
            const int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx / pad_period * pad;
            gmem[i * N + j] = __low2half(smem[smem_idx]);
        }
    }
}

__global__ void permute_filter_layout4096(const cpx_h* __restrict__ d_H_in,
                                          cpx_h* __restrict__ d_H_perm,
                                          int D) {
    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;
    static_assert(BPB == 1, "permute_filter_layout4096 assumes BPB == 1");

    const int d = blockIdx.x;
    if (d >= D) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_linear = ty * thunderfft::threads_per_warp + tx;
    const int h_base = d * kN;
    const int out_base = d * kN;
    const int col_base = (tx >> 2) + (ty << 4);
    const int row_base = (tx & 3) << 4;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int idx0 = h_base + (col_base * 64) + row_base + i;
        const int idx1 = h_base + ((col_base + 8) * 64) + row_base + i;
        d_H_perm[out_base + i * block_threads + thread_linear] = d_H_in[idx0];
        d_H_perm[out_base + (i + 16) * block_threads + thread_linear] = d_H_in[idx1];
    }
}

__global__ void thunder_conv4096_block_kernel(const cpx_h* __restrict__ d_input,
                                              const cpx_h* __restrict__ d_H_perm,
                                              cpx_h* __restrict__ d_output,
                                              int D) {
    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    constexpr int ept = kN * BPB / (thunderfft::threads_per_warp * WPB);
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, kN, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, kN, BPB>::L_out;

    cpx_h W_fwd[36];
    cpx_h W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kN, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<kN, false>(W_inv);

    static_assert(BPB == 1, "thunder_conv4096_block_kernel assumes BPB == 1");
    static_assert(ept == 32, "thunder_conv4096_block_kernel assumes ept == 32");
    const int d = blockIdx.x % D;
    const int h_base = d * kN;
    const int thread_linear = threadIdx.y * thunderfft::threads_per_warp + threadIdx.x;
    thunderfft::ThunderFFT_gmem2smem<half, L_in>(s_in, d_input + blockIdx.x * BPB * kN);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, true>(reg, W_fwd, s_in);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int idx = h_base + i * block_threads + thread_linear;
        reg[i] = cmul_h2(reg[i], d_H_perm[idx]);
    }

    reverse_data(reg);

    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, false>(reg, W_inv, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    thunderfft::ThunderFFT_smem2gmem<half, L_out>(d_output + blockIdx.x * BPB * kN, s_in);
}

__global__ void thunder_conv4096_block_kernel_realin(const half* __restrict__ d_input_real,
                                                     const cpx_h* __restrict__ d_H_perm,
                                                     cpx_h* __restrict__ d_output,
                                                     int D) {
    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    constexpr int ept = kN * BPB / (thunderfft::threads_per_warp * WPB);
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, kN, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, kN, BPB>::L_out;

    cpx_h W_fwd[36];
    cpx_h W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kN, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<kN, false>(W_inv);

    static_assert(BPB == 1, "thunder_conv4096_block_kernel_realin assumes BPB == 1");
    static_assert(ept == 32, "thunder_conv4096_block_kernel_realin assumes ept == 32");
    const int d = blockIdx.x % D;
    const int h_base = d * kN;
    const int thread_linear = threadIdx.y * thunderfft::threads_per_warp + threadIdx.x;

    ThunderFFT_gmem2smem_realio<L_in>(s_in, d_input_real + blockIdx.x * BPB * kN);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, true>(reg, W_fwd, s_in);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int idx = h_base + i * block_threads + thread_linear;
        reg[i] = cmul_h2(reg[i], d_H_perm[idx]);
    }

    reverse_data(reg);

    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, false>(reg, W_inv, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    thunderfft::ThunderFFT_smem2gmem<half, L_out>(d_output + blockIdx.x * BPB * kN, s_in);
}

__global__ void thunder_conv4096_block_kernel_realin_realout(const half* __restrict__ d_input_real,
                                                             const cpx_h* __restrict__ d_H_perm,
                                                             half* __restrict__ d_output_real,
                                                             int D) {
    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    constexpr int ept = kN * BPB / (thunderfft::threads_per_warp * WPB);
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, kN, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, kN, BPB>::L_out;

    cpx_h W_fwd[36];
    cpx_h W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kN, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<kN, false>(W_inv);

    static_assert(BPB == 1, "thunder_conv4096_block_kernel_realin_realout assumes BPB == 1");
    static_assert(ept == 32, "thunder_conv4096_block_kernel_realin_realout assumes ept == 32");
    const int d = blockIdx.x % D;
    const int h_base = d * kN;
    const int thread_linear = threadIdx.y * thunderfft::threads_per_warp + threadIdx.x;

    ThunderFFT_gmem2smem_realio<L_in>(s_in, d_input_real + blockIdx.x * BPB * kN);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, true>(reg, W_fwd, s_in);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int idx = h_base + i * block_threads + thread_linear;
        reg[i] = cmul_h2(reg[i], d_H_perm[idx]);
    }

    reverse_data(reg);

    thunderfft::ThunderFFT_kernel_reg<half, kN, BPB, false>(reg, W_inv, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    ThunderFFT_smem2gmem_realio<L_out>(d_output_real + blockIdx.x * BPB * kN, s_in);
}


__global__ void f32_to_h2(const float2* __restrict__ in,
                          half2* __restrict__ out,
                          int n) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n) out[i] = __floats2half2_rn(in[i].x, in[i].y);
}

}  // namespace

int main(int argc, char** argv) {
    int K = 9;
    int device = 0;
    int B = 1;
    int D = 1;
    int warmup = 5;
    int runs = 30;
    int include_filter_fft = 1;
    int validate = 1;
    int filter_fft_mode = 1; // 0: cuFFT, 1: ThunderFFT
    int real_input = 0;
    int real_filter = 0;

    if (argc >= 2) K = std::atoi(argv[1]);
    if (argc >= 3) device = std::atoi(argv[2]);
    if (argc >= 4) B = std::atoi(argv[3]);
    if (argc >= 5) D = std::atoi(argv[4]);
    if (argc >= 6) warmup = std::atoi(argv[5]);
    if (argc >= 7) runs = std::atoi(argv[6]);
    if (argc >= 8) include_filter_fft = std::atoi(argv[7]);
    if (argc >= 9) validate = std::atoi(argv[8]);
    if (argc >= 10) filter_fft_mode = std::atoi(argv[9]);
    if (argc >= 11) real_input = std::atoi(argv[10]);
    if (argc >= 12) real_filter = std::atoi(argv[11]);

    if (K < 1 || K > kN) {
        std::cerr << "K must be in [1, " << kN << "]\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    std::srand(42);

    const int batch_count = B * D;
    const int sig_elems = batch_count * kN;
    const int filt_elems = D * kN;

    std::vector<cpx_h> h_x(sig_elems);
    std::vector<half> h_x_real(sig_elems, __float2half(0.0f));
    std::vector<float2> h_h_time(filt_elems, make_float2(0.f, 0.f));
    std::vector<cpx_h> h_h_half(filt_elems, __float2half2_rn(0.0f));

    for (int i = 0; i < sig_elems; ++i) {
        float re = std::rand() / float(RAND_MAX);
        float im = real_input ? 0.0f : (std::rand() / float(RAND_MAX));
        h_x[i] = __floats2half2_rn(re, im);
        h_x_real[i] = __float2half(re);
    }
    for (int d = 0; d < D; ++d) {
        for (int k = 0; k < K; ++k) {
            // Scale filter by 1/N so inverse (unnormalized) recovers linear amplitude
            // and to avoid half overflow in frequency-domain product.
            float re = (std::rand() / float(RAND_MAX)) / float(kN);
            float im = real_filter ? 0.0f : ((std::rand() / float(RAND_MAX)) / float(kN));
            h_h_time[d * kN + k] = make_float2(re, im);
            h_h_half[d * kN + k] = __floats2half2_rn(re, im);
        }
    }

    cpx_h *d_x = nullptr, *d_out = nullptr, *d_H = nullptr, *d_H_perm = nullptr, *d_h_half = nullptr;
    half *d_x_real = nullptr, *d_out_real = nullptr;
    float2 *d_h_time = nullptr, *d_h_freq = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, sizeof(cpx_h) * sig_elems));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(cpx_h) * sig_elems));
    CHECK_CUDA(cudaMalloc(&d_H, sizeof(cpx_h) * filt_elems));
    CHECK_CUDA(cudaMalloc(&d_H_perm, sizeof(cpx_h) * filt_elems));
    CHECK_CUDA(cudaMalloc(&d_h_half, sizeof(cpx_h) * filt_elems));
    CHECK_CUDA(cudaMalloc(&d_h_time, sizeof(float2) * filt_elems));
    CHECK_CUDA(cudaMalloc(&d_h_freq, sizeof(float2) * filt_elems));
    if (real_input) {
        CHECK_CUDA(cudaMalloc(&d_x_real, sizeof(half) * sig_elems));
        CHECK_CUDA(cudaMemcpy(d_x_real, h_x_real.data(), sizeof(half) * sig_elems, cudaMemcpyHostToDevice));
    }
    if (real_input && real_filter) {
        CHECK_CUDA(cudaMalloc(&d_out_real, sizeof(half) * sig_elems));
    }

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), sizeof(cpx_h) * sig_elems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_time, h_h_time.data(), sizeof(float2) * filt_elems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_half, h_h_half.data(), sizeof(cpx_h) * filt_elems, cudaMemcpyHostToDevice));

    cufftHandle plan_h;
    CHECK_CUFFT(cufftPlan1d(&plan_h, kN, CUFFT_C2C, D));
    thunderfft::ThunderFFTInitialize<half>(kN);

    constexpr int BPB = thunderfft::batch_per_block<kN>;
    constexpr int WPB = thunderfft::warp_per_block<kN>;
    const dim3 grid_signal(batch_count / BPB);
    const dim3 grid_filter(D / BPB);
    const dim3 block(thunderfft::threads_per_warp, WPB);
    const size_t shmem_bytes = sizeof(cpx_h) * (kN + pad_h(kN)) * BPB;
    CHECK_CUDA(cudaFuncSetAttribute(
        thunder_fft4096_block_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(
        thunder_conv4096_block_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(
        thunder_conv4096_block_kernel_realin,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(
        thunder_conv4096_block_kernel_realin_realout,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));

    auto filter_fft_once = [&]() {
        if (filter_fft_mode == 0) {
            CHECK_CUFFT(cufftExecC2C(plan_h,
                                     reinterpret_cast<cufftComplex*>(d_h_time),
                                     reinterpret_cast<cufftComplex*>(d_h_freq),
                                     CUFFT_FORWARD));
            f32_to_h2<<<(filt_elems + 255) / 256, 256>>>(d_h_freq, d_H, filt_elems);
            CHECK_CUDA(cudaGetLastError());
        } else {
            thunder_fft4096_block_kernel<true><<<grid_filter, block, shmem_bytes>>>(d_h_half, d_H);
            CHECK_CUDA(cudaGetLastError());
        }
        permute_filter_layout4096<<<D, block>>>(d_H, d_H_perm, D);
        CHECK_CUDA(cudaGetLastError());
    };

    if (!include_filter_fft) {
        filter_fft_once();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto run_once = [&]() {
        if (include_filter_fft) {
            filter_fft_once();
        }
        if (real_input) {
            if (real_filter) {
                thunder_conv4096_block_kernel_realin_realout<<<grid_signal, block, shmem_bytes>>>(
                    d_x_real, d_H_perm, d_out_real, D);
            } else {
                thunder_conv4096_block_kernel_realin<<<grid_signal, block, shmem_bytes>>>(
                    d_x_real, d_H_perm, d_out, D);
            }
        } else {
            thunder_conv4096_block_kernel<<<grid_signal, block, shmem_bytes>>>(d_x, d_H_perm, d_out, D);
        }
        CHECK_CUDA(cudaGetLastError());
    };

    float ms = measure_execution_ms(run_once, unsigned(warmup), unsigned(runs));
    std::cout << std::fixed << std::setprecision(6)
              << "ThunderFFT full FFT-Conv (N=4096, include_filter_fft="
              << include_filter_fft << ", filter_fft_mode=" << filter_fft_mode
              << ") time: " << ms << " ms\n";

    if (validate) {
        std::vector<cpx_h> h_out(sig_elems);
        if (real_input && real_filter) {
            std::vector<half> h_out_real(sig_elems);
            CHECK_CUDA(cudaMemcpy(h_out_real.data(), d_out_real, sizeof(half) * sig_elems, cudaMemcpyDeviceToHost));
            for (int i = 0; i < sig_elems; ++i) h_out[i] = __halves2half2(h_out_real[i], __float2half(0.0f));
        } else {
            CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(cpx_h) * sig_elems, cudaMemcpyDeviceToHost));
        }

        double max_abs_raw = 0.0;
        double max_abs_scaled = 0.0;
        double num_raw = 0.0, num_scaled = 0.0, den = 0.0;
        double max_abs_scaled_corr = 0.0;
        double num_scaled_corr = 0.0, den_corr = 0.0;
        for (int bd = 0; bd < batch_count; ++bd) {
            const int d = bd % D;
            for (int n = 0; n < kN; ++n) {
                float ref_re = 0.0f, ref_im = 0.0f;
                float refc_re = 0.0f, refc_im = 0.0f;
                for (int kk = 0; kk < K; ++kk) {
                    const int idx = n - kk;
                    const int xi = (idx >= 0) ? idx : (idx + kN);
                    float2 xv = __half22float2(h_x[bd * kN + xi]);
                    float2 hv = h_h_time[d * kN + kk];
                    ref_re += xv.x * hv.x - xv.y * hv.y;
                    ref_im += xv.x * hv.y + xv.y * hv.x;

                    const int idxc = n + kk;
                    const int xic = (idxc < kN) ? idxc : (idxc - kN);
                    float2 xvc = __half22float2(h_x[bd * kN + xic]);
                    refc_re += xvc.x * hv.x - xvc.y * hv.y;
                    refc_im += xvc.x * hv.y + xvc.y * hv.x;
                }
                float2 ov = __half22float2(h_out[bd * kN + n]);
                const float dx_raw = ov.x - ref_re;
                const float dy_raw = ov.y - ref_im;
                const float dx_scaled = ov.x / float(kN) - ref_re;
                const float dy_scaled = ov.y / float(kN) - ref_im;
                const double abs_raw = std::sqrt(double(dx_raw) * dx_raw + double(dy_raw) * dy_raw);
                const double abs_scaled = std::sqrt(double(dx_scaled) * dx_scaled + double(dy_scaled) * dy_scaled);
                max_abs_raw = std::max(max_abs_raw, abs_raw);
                max_abs_scaled = std::max(max_abs_scaled, abs_scaled);
                num_raw += double(dx_raw) * dx_raw + double(dy_raw) * dy_raw;
                num_scaled += double(dx_scaled) * dx_scaled + double(dy_scaled) * dy_scaled;
                den += double(ref_re) * ref_re + double(ref_im) * ref_im;

                const float dx_corr = ov.x / float(kN) - refc_re;
                const float dy_corr = ov.y / float(kN) - refc_im;
                const double abs_corr = std::sqrt(double(dx_corr) * dx_corr + double(dy_corr) * dy_corr);
                max_abs_scaled_corr = std::max(max_abs_scaled_corr, abs_corr);
                num_scaled_corr += double(dx_corr) * dx_corr + double(dy_corr) * dy_corr;
                den_corr += double(refc_re) * refc_re + double(refc_im) * refc_im;
            }
        }
        const double l2_raw = den > 0.0 ? std::sqrt(num_raw / den) : std::sqrt(num_raw);
        const double l2_scaled = den > 0.0 ? std::sqrt(num_scaled / den) : std::sqrt(num_scaled);
        const double l2_scaled_corr =
            den_corr > 0.0 ? std::sqrt(num_scaled_corr / den_corr) : std::sqrt(num_scaled_corr);
        std::cout << std::setprecision(6)
                  << "Validation vs direct circular conv (raw): L2_rel=" << l2_raw
                  << " max_abs=" << max_abs_raw << "\n"
                  << "Validation vs direct circular conv (raw/N): L2_rel=" << l2_scaled
                  << " max_abs=" << max_abs_scaled << "\n"
                  << "Validation vs direct circular corr (raw/N): L2_rel=" << l2_scaled_corr
                  << " max_abs=" << max_abs_scaled_corr << "\n";
    }

    thunderfft::ThunderFFTFinalize<half>();
    CHECK_CUFFT(cufftDestroy(plan_h));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_H));
    CHECK_CUDA(cudaFree(d_H_perm));
    CHECK_CUDA(cudaFree(d_h_half));
    CHECK_CUDA(cudaFree(d_h_time));
    CHECK_CUDA(cudaFree(d_h_freq));
    if (d_x_real) CHECK_CUDA(cudaFree(d_x_real));
    if (d_out_real) CHECK_CUDA(cudaFree(d_out_real));
    return 0;
}
