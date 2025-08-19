# === 설정 ===
NVCC        := nvcc
TARGET      := fft
SRC_FILES   := main.cu baseline_fft.cu my_fft.cu
CUFFT_LIBS  := -lcufft
ARCH_FLAGS  := -gencode arch=compute_70,code=sm_70

# === 기본 빌드 ===
all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(NVCC) -O3 -std=c++17 $(ARCH_FLAGS) $(SRC_FILES) -o $(TARGET) $(CUFFT_LIBS)

# === 청소 ===
clean:
	rm -f $(TARGET)
