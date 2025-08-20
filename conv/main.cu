#include "convolution.hpp"
#include <getopt.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool validate(float *ref, float *target, int size, const char *tag);

void initialize_with_gaussian(float *data, int size, float mean = 0.0f,
                              float stddev = 0.3f) {
  std::default_random_engine gen;
  std::normal_distribution<float> dist(mean, stddev);
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }
}

int main(int argc, char **argv) {
  int N = 4;
  int f = 2;
  int T = 16; // default tile size

  // Parse arguments
  static struct option long_options[] = {{"N", required_argument, 0, 0},
                                         {"f", required_argument, 0, 0},
                                         {"T", required_argument, 0, 0},
                                         {0, 0, 0, 0}};

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1)
      break;
    if (strcmp(long_options[option_index].name, "N") == 0)
      N = atoi(optarg);
    else if (strcmp(long_options[option_index].name, "f") == 0)
      f = atoi(optarg);
    else if (strcmp(long_options[option_index].name, "T") == 0)
      T = atoi(optarg);
  }

  if (f > N) {
    printf("Filter size f must be <= N\n");
    return 1;
  }

  int out_size = N - f + 1;

  float *input = (float *)malloc(sizeof(float) * N * N);
  float *filter = (float *)malloc(sizeof(float) * f * f);
  float *output_naive = (float *)malloc(sizeof(float) * out_size * out_size);
  float *output_cudnn = (float *)malloc(sizeof(float) * out_size * out_size);
  float *output_cufft = (float *)malloc(sizeof(float) * out_size * out_size);
  float *output_cufft_tiled =
      (float *)malloc(sizeof(float) * out_size * out_size);
  float *output_cufftdx = (float *)malloc(sizeof(float) * out_size * out_size);

  initialize_with_gaussian(input, N * N);
  initialize_with_gaussian(filter, f * f);

  printf("Running convolution with N=%d, f=%d, T=%d\n", N, f, T);

  // Each function handles its own timing & FLOP reporting
  // printf("running naive\n");
  // convolution_naive(input, filter, output_naive, N, f);
  // convolution_cudnn(input, filter, output_cudnn, N, f);
  convolution_cufft(input, filter, output_cufft, N, f);
  convolution_cufft_tiled(input, filter, output_cufft_tiled, N, f, T);
  // for(int i=0;i<100;i++)
  convolution_cufftdx(input, filter, output_cufftdx, N, f); // tiled

  // Validation
  // validate(output_naive, output_cudnn, out_size * out_size, "cuDNN");
  // validate(output_cudnn, output_cufft, out_size * out_size, "cuDNN vs
  // cuFFT");
  validate(output_cufft, output_cufft_tiled, out_size * out_size,
           "cuFFT vs cuFFT Tiled");
  validate(output_cufft, output_cufftdx, out_size * out_size,
           "cuFFT vs cuFFTdx");

  // for(int i=0; i<out_size; i++) {
  //     for(int j=0; j<out_size; j++)
  //         printf("%f ", output_cudnn[i*out_size+j]);
  //     printf("\n");
  // }
  // printf("------\n");
  // for(int i=0; i<out_size; i++) {
  //     for(int j=0; j<out_size; j++)
  //         printf("%f ", output_cufft_tiled[i*out_size+j]);
  //     printf("\n");
  // }
  // printf("\n\n\n");
  // for(int i=0; i<out_size; i++){
  //   for(int j=0; j<out_size; j++)
  //     printf("%f ", output_cufftdx[i*out_size+j]);
  //   printf("\n");
  // }
  free(input);
  free(filter);
  free(output_naive);
  free(output_cudnn);
  free(output_cufft);

  return 0;
}
