cmake_minimum_required(VERSION 3.10)
if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
project(SGEMM_CUDA CUDA CXX)

# Find CUDA package.
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

# Include directories.
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDA_INCLUDE_DIRS})

# Add executables
cuda_add_executable(test_correctness
    src/test_correctness.cu
    src/common.cuh
)
cuda_add_executable(test_performance
    src/test_performance.cu
    src/common.cuh
)

# Link CUDA libraries.
target_link_libraries(test_correctness ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(test_performance ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
