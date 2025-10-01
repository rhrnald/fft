# Library: thunderfft (host wrapper + device core)
add_library(thunderfft STATIC
  ${PROJECT_SOURCE_DIR}/src/thunderfft.cu
)

target_include_directories(thunderfft
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable separable compilation if requested
set_target_properties(thunderfft PROPERTIES
  CUDA_SEPARABLE_COMPILATION ${THUNDERFFT_ENABLE_RDC}
)

# Common compile options
target_compile_options(thunderfft
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(thunderfft
  PUBLIC
    CUDA::cudart
)

