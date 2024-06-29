#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace v2 {

// Let's keep everything quadratic for simplicity now, but I can see the
// optimal values being different between their A and B counterparts.
//
// This is the normal block size, but each thread now computes multiple values.
constexpr uint32_t BLOCK_SIZE = 32;
// Inner dimension of the chunk that we load into SMEM.
// Its other dimension is BLOCK_SIZE * TILE_SIZE.
constexpr uint32_t CHUNK_SIZE = 32;
// I actually have no idea how this one is called properly.
// This is the size of the innermost loop of each thread.
constexpr uint32_t TILE_SIZE = 4;
//
// Illustration:
//
//                BLOCK=2
//               /      /
//             ____   ____
//
//             TILE=4
//               |
//               /
//           [ [XXYY] [ZZDD]  ...]
//           [ [XXYY] [ZZDD]  ...]
//  CHUNK=2  [                ...]
//     |     [                ...]
//    /      [                ...]
//            +------+------+
// [[XX]    ] | 0000 | 2222 |
// [[XX]    ] | 0000 | 2222 |
// [[YY]    ] | 0000 | 2222 |
// [[YY]    ] | 0000 | 2222 |
//            +------+------+
// [[ZZ]    ] | 1111 | 3333 |
// [[ZZ]    ] | 1111 | 3333 |
// [[DD]    ] | 1111 | 3333 |
// [[DD]    ] | 1111 | 3333 |
//            +------+------+
// [........]
// [........]
//
// Tile is the innermost block computed inside each thread. In the picture
// above, there are four tiles: one made of 0's, one of 1's, etc. Each one is
// 4x4, which is specified by TILE_SIZE.
//
// Chunk is the thing that we load into SMEM. Each thread loads its part of the
// chunk. There's an A chunk and a B chunk. Which thread loads which is denoted
// with uppercase letters: thread 0 loads X, thread 1 loads Y, etc.
// 
// Block is the same thing here and in the previous kernels, although now that
// each thread computes a 4x4 tile, the effective size of the C submatrix that
// each block computes is BLOCK_SIZE * TILE_SIZE = 8.


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// SGEMM kernel implementation.
__global__ void sgemm_kernel(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
) {
    // TODO.
}

void sgemm(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
) {
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 dimGrid(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    sgemm_kernel<<<dimGrid, dimBlock>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}

} // namespace v2
