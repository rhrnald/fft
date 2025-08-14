#include <math.h>
#include "my_fft.h"

void baseline_fft(float2* d_data, int N);
void my_fft_original(float2* d_data, int N);

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
    const int N = 64*16*65536;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Max grid size: x=%d, y=%d, z=%d\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);
    float2* h_input = (float2*)malloc(sizeof(float2) * N);
    for (int i = 0; i < N; ++i) {
        // h_input[i].x = sinf(2 * M_PI * i / 64); // real part
        // h_input[i].y = 0.0f;                   // imag part
        h_input[i].x=i%64;
        h_input[i].y=0;
    }

    float2 *d_baseline, *d_custom;
    cudaMalloc(&d_baseline, sizeof(float2) * N);
    cudaMalloc(&d_custom, sizeof(float2) * N);

    cudaMemcpy(d_baseline, h_input, sizeof(float2) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_custom, h_input, sizeof(float2) * N, cudaMemcpyHostToDevice);

    baseline_fft(d_baseline, N);
    // my_fft_original(d_custom, N);
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

