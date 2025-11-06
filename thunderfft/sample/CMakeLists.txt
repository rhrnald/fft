# Sample executable that includes <thunderfft/thunderfft.h>
add_executable(thff_fft
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cu
)

add_executable(thff_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cu
)

# Link header-only library (exports include dirs + cudart)
target_link_libraries(thff_fft PRIVATE thunderfft CUDA::cudart CUDA::cufft)
target_link_libraries(thff_conv PRIVATE thunderfft CUDA::cudart CUDA::cufft)

target_include_directories(thff_conv PRIVATE /home/sumin/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/external/cutlass/include)
target_include_directories(thff_conv PRIVATE /home/sumin/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include)

# Optional: common flags
target_compile_options(thff_fft PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(thff_conv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)