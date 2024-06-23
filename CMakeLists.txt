cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(SGEMM_CUDA CUDA CXX)

# Find CUDA package.
find_package(CUDA REQUIRED)

# Include directories.
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add executable
cuda_add_executable(test_correctness
    src/test_correctness.cu
    src/sgemm/v1_simple.cuh
)

# Link CUDA libraries.
target_link_libraries(test_correctness ${CUDA_LIBRARIES})
