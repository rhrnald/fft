# Sample executable that includes <thunderfft/thunderfft.h>
add_executable(thff_sample
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
)

# Link header-only library (exports include dirs + cudart)
target_link_libraries(thff_sample PRIVATE thunderfft CUDA::cudart CUDA::cufft)

# Optional: common flags
target_compile_options(thff_sample PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
