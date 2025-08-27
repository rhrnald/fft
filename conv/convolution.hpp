#pragma once

void convolution_naive(float* input, float* filter, float* output, int N, int f);
void convolution_cudnn(float* h_input, float* h_filter, float* h_output, int N, int f);
void convolution_cufft(float* h_input, float* h_filter, float* h_output, int N, int f);
// void convolution_cufft_tiled(float* h_input, float* h_filter, float* h_output, int N, int f, int tile_size);
void convolution_cufftdx(float* h_input, float* h_filter, float* h_output, int N, int f);
void my_convolution(float* h_input, float* h_filter, float* h_output, int N, int f);
