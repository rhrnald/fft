#include <math.h>
#include <stdio.h>

#include "stat.h"
#include "utils.h"

#include "my_fft.h"
#include "fft_tc_sm_bench.h"
#include "my_fft.h"


void baseline_fft(float2 *h_input, float2 *h_output, int N, int batch);


int main() {
    constexpr long long N = 64;
    constexpr long long batch = 65536;
    
    float2 *h_input = (float2 *)malloc(sizeof(float2) * N * batch);
    half2 *h_input_half = (half2 *)malloc(sizeof(half2) * N * batch);
    float2 *h_answer = (float2 *)malloc(sizeof(float2) * N * batch);

    for (int i = 0; i < N * batch; ++i) {
        h_input[i].x = sinf(2 * M_PI * (i % N) / 64);
        // h_input[i].x = i % N;
        h_input[i].y = 0.0f;

        h_input_half[i] = make_half2(h_input[i].x, h_input[i].y);
    }

    baseline_fft(h_input, h_answer, N, batch);

    float2 *d_input;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float2) * N * batch));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float2) * N * batch, cudaMemcpyHostToDevice));

    my_fft_benchmark<N>(h_input, h_input_half, h_answer, batch);
    // stat::print_table();

    fft_tc_sm_benchmark<N>(h_input, h_input_half, h_answer, batch);
    // stat::print_table();

    // fft_tc_sm_benchmark<256>(h_input, h_input_half, answer, batch);
    

    
    stat::set_title("FFT benchmark results");
    stat::print_table();
    return 0;
}
