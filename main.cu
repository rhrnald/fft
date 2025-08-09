#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                      \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

void baseline_fft(float2* d_data, int N);
template<int N> void my_fft(float2* d_data);

void check_result(float2* ref, float2* test, int N) {
    float max_err = 0.0f;
    
    // for(int i=0; i<N; i++) printf("(%f + %f i) ", ref[i].x, ref[i].y); printf("\n"); for(int i=0; i<N; i++) printf("(%f + %f i) ", test[i].x, test[i].y); printf("\n");

    for (int i = 0; i < N; ++i) {
        float dx = ref[i].x - test[i].x;
        float dy = ref[i].y - test[i].y;
        float err = sqrtf(dx * dx + dy * dy);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);
}

int main() {
    const int N = 1024;
    float2* h_input = (float2*)malloc(sizeof(float2) * N);
    for (int i = 0; i < N; ++i) {
        // h_input[i].x = sinf(2 * M_PI * i / N); // real part
        // h_input[i].y = 0.0f;                   // imag part
        h_input[i].x=i;
        h_input[i].y=i;
    }

    float2 *d_baseline, *d_custom;
    cudaMalloc(&d_baseline, sizeof(float2) * N);
    cudaMalloc(&d_custom, sizeof(float2) * N);

    cudaMemcpy(d_baseline, h_input, sizeof(float2) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_custom, h_input, sizeof(float2) * N, cudaMemcpyHostToDevice);

    baseline_fft(d_baseline, N);
    my_fft<N>(d_custom);

    float2* h_baseline = (float2*)malloc(sizeof(float2) * N);
    float2* h_custom = (float2*)malloc(sizeof(float2) * N);
    cudaMemcpy(h_baseline, d_baseline, sizeof(float2) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_custom, d_custom, sizeof(float2) * N, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++) if(abs(h_baseline[i].x-h_custom[i].x) > 1e-9 || abs(h_baseline[i].y-h_custom[i].y) > 1e-9) printf("(%f + %f i / %f + %f i \n) ", h_baseline[i].x, h_baseline[i].y, h_custom[i].x, h_custom[i].y);
    check_result(h_baseline, h_custom, N);

    free(h_input);
    free(h_baseline);
    free(h_custom);
    cudaFree(d_baseline);
    cudaFree(d_custom);
    return 0;
}

