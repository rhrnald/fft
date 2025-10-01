# Benchmark executable (add/adjust sources as you have them)
add_executable(thff_bench
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  # ${PROJECT_SOURCE_DIR}/bench/baseline_cufft.cu
  # ${PROJECT_SOURCE_DIR}/bench/my_fft.cu
  # ${PROJECT_SOURCE_DIR}/bench/stat.cu
)

target_link_libraries(thff_bench PRIVATE
  thunderfft        # header-only lib (includes cudart)
  CUDA::cufft       # for cuFFT baseline, comment out if unused
)

target_compile_options(thff_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

