# Makefile for cuFFT, cuDNN, cuFFTDx convolution comparison
HOME_DIR = /home/chaewon/
ARCH_NO=80

# Paths
MATHDX_ROOT = $(HOME_DIR)/install/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06

# Compiler and target
NVCC = nvcc
TARGET = conv_compare

# Source files
SRCS = main.cu conv_naive.cpp cudnn.cu cufft.cu cufftdx.cu myconv.cu
OBJS = $(SRCS:.cu=.o)

# Flags
ARCH = sm_$(ARCH_NO)
NVCC_FLAGS = -std=c++17 -O3 -arch=$(ARCH) -DCUFFT_TARGET_ARCHS=$(ARCH_NO)0
CXXFLAGS := -O3 -std=c++14 -Wall -march=native -mavx2 -mno-avx512f -mfma \
            -fopenmp -I/usr/local/cuda/include -fopenmp

INCLUDES = -I. -I$(MATHDX_ROOT)/include -I$(MATHDX_ROOT)/external/cutlass/include/
LIBS = -lcudart -lcudnn -lcufft

# Build rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET) $(TARGET)_fftdx *.o
