#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#ifdef FLOAT_AS_DOUBLE
typedef double Float;
#define CUDA_FLOAT_TYPE CUDA_R_64F
#else
typedef float Float;
#define CUDA_FLOAT_TYPE CUDA_R_32F
#endif

extern cublasHandle_t handle;

// Thread block dimensions for matrix operations
constexpr int TILE_DIM = 16;
constexpr int BLOCK_SIZE = 256;

/**
 * Fused kernel: D = alpha * A @ (beta * B + gamma * C)^T + out_coef * D
 * 
 * This kernel computes the matrix multiplication of A with the transpose
 * of a linear combination of B and C, all in a single operation.
 * 
 * Parameters:
 * - A: Input matrix A (M x K)
 * - B: Input matrix B (N x K) 
 * - C: Input matrix C (N x K)
 * - D: Output matrix D (M x N)
 * - alpha: Scalar multiplier for the entire operation
 * - beta: Scalar multiplier for matrix B
 * - gamma: Scalar multiplier for matrix C
 */
__global__ void fused_A_mul_B_plus_C_transpose_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B, const Float* __restrict__ C, Float* __restrict__ D,
    int lda, int ldb, int ldc, int ldd, int M, int N, int K,
    Float alpha, Float beta, Float gamma, Float out_coef) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        Float sum = 0.0;
        
        // Compute D[row][col] = out_coef * D[row][col] + alpha * sum_k(A[row][k] * (beta * B[col][k] + gamma * C[col][k]))
        for (int k = 0; k < K; k++) {
            Float a_val = A[row + k * lda];
            Float bc_val = beta * B[col + k * ldb] + gamma * C[col + k * ldc];
            sum += a_val * bc_val;
        }
        
        D[row + col * ldd] = out_coef * D[row + col * ldd] + alpha * sum;
    }
}

/**
 * Optimized tiled version using shared memory
 */
template<int TileDim>
__global__ void fused_A_mul_B_plus_C_transpose_tiled_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B, const Float* __restrict__ C, Float* __restrict__ D,
    int lda, int ldb, int ldc, int ldd, int M, int N, int K,
    Float alpha, Float beta, Float gamma, Float out_coef) {
    
    __shared__ Float tile_A[TILE_DIM][TILE_DIM];
    __shared__ Float tile_BC[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    Float sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        // Load tile of A
        int a_row = row;
        int a_col = tile * TILE_DIM + threadIdx.x;
        if (a_row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[a_row + a_col * lda];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load and compute tile of (beta * B + gamma * C)
        int bc_row = col;
        int bc_col = tile * TILE_DIM + threadIdx.y;
        if (bc_row < N && bc_col < K) {
            Float b_val = B[bc_row + bc_col * ldb];
            Float c_val = C[bc_row + bc_col * ldc];
            tile_BC[threadIdx.x][threadIdx.y] = beta * b_val + gamma * c_val;
        } else {
            tile_BC[threadIdx.x][threadIdx.y] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_DIM; k++) {
            sum += tile_A[threadIdx.y][k] * tile_BC[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        D[row + col * ldd] = out_coef * D[row + col * ldd] + alpha * sum;
    }
}

/**
 * Fused kernel: D = alpha * (beta * A + gamma * B) @ C^T
 * 
 * This kernel computes the matrix multiplication of a linear combination
 * of A and B with the transpose of C, all in a single operation.
 * 
 * Parameters:
 * - A: Input matrix A (M x K)
 * - B: Input matrix B (M x K)
 * - C: Input matrix C (N x K)
 * - D: Output matrix D (M x N)
 * - alpha: Scalar multiplier for the entire operation
 * - beta: Scalar multiplier for matrix A
 * - gamma: Scalar multiplier for matrix B
 */
__global__ void fused_A_plus_B_mul_C_transpose_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B, const Float* __restrict__ C, Float* __restrict__ D,
    int lda, int ldb, int ldc, int ldd, int M, int N, int K,
    Float alpha, Float beta, Float gamma, Float out_coef) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        Float sum = 0.0;
        
        // Compute D[row][col] = out_coef * D[row][col] + alpha * sum_k((beta * A[row][k] + gamma * B[row][k]) * C[col][k])
        for (int k = 0; k < K; k++) {
            Float ab_val = beta * A[row + k * lda] + gamma * B[row + k * ldb];
            Float c_val = C[col + k * ldc];
            sum += ab_val * c_val;
        }
        
        D[row + col * ldd] = out_coef * D[row + col * ldd] + alpha * sum;
    }
}

/**
 * Optimized tiled version using shared memory
 */
template<int TileDim>
__global__ void fused_A_plus_B_mul_C_transpose_tiled_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B, const Float* __restrict__ C, Float* __restrict__ D,
    int lda, int ldb, int ldc, int ldd, int M, int N, int K,
    Float alpha, Float beta, Float gamma, Float out_coef) {
    
    __shared__ Float tile_AB[TILE_DIM][TILE_DIM];
    __shared__ Float tile_C[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    Float sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        // Load and compute tile of (beta * A + gamma * B)
        int ab_row = row;
        int ab_col = tile * TILE_DIM + threadIdx.x;
        if (ab_row < M && ab_col < K) {
            Float a_val = A[ab_row + ab_col * lda];
            Float b_val = B[ab_row + ab_col * ldb];
            tile_AB[threadIdx.y][threadIdx.x] = beta * a_val + gamma * b_val;
        } else {
            tile_AB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile of C
        int c_row = col;
        int c_col = tile * TILE_DIM + threadIdx.y;
        if (c_row < N && c_col < K) {
            tile_C[threadIdx.x][threadIdx.y] = C[c_row + c_col * ldc];
        } else {
            tile_C[threadIdx.x][threadIdx.y] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_DIM; k++) {
            sum += tile_AB[threadIdx.y][k] * tile_C[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        D[row + col * ldd] = out_coef * D[row + col * ldd] + alpha * sum;
    }
}

/**
 * Kernel for computing two output matrices from two input matrices:
 * C1 = gamma * (alpha * A + beta * B)
 * C2 = delta * (alpha * A + beta * B)
 * 
 * This kernel allows computing a linear combination once and storing it in two
 * different locations with different coefficients.
 * 
 * Parameters:
 * - A: First input matrix (M x N)
 * - B: Second input matrix (M x N)
 * - C1: First output matrix (M x N)
 * - C2: Second output matrix (M x N)
 * - alpha: Coefficient for matrix A
 * - beta: Coefficient for matrix B
 * - gamma: Coefficient for first output
 * - delta: Coefficient for second output
 * - out_coef1: Coefficient for previously stored value of the first output
 * - out_coef2: Coefficient for previously stored value of the second output
 */
__global__ void sum_to_2_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B,
    Float* __restrict__ C1, Float* __restrict__ C2,
    int lda, int ldb, int ldc1, int ldc2, int M, int N,
    Float alpha, Float beta, Float gamma, Float delta,
    Float out_coef1, Float out_coef2) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Compute linear combination once
        Float sum = alpha * A[row + col * lda] + beta * B[row + col * ldb];
        
        // Store in both output locations with respective coefficients and accumulation
        C1[row + col * ldc1] = out_coef1 * C1[row + col * ldc1] + gamma * sum;
        C2[row + col * ldc2] = out_coef2 * C2[row + col * ldc2] + delta * sum;
    }
}

/**
 * Host wrapper for computing two output matrices from two input matrices
 */
void GPU_sum_to_2(Float *A, Float *B, Float *C1, Float *C2,
                  int lda, int ldb, int ldc1, int ldc2,
                  int M, int N,
                  Float alpha, Float beta, Float gamma, Float delta,
                  Float out_coef1, Float out_coef2) {
    
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    sum_to_2_kernel<<<gridSize, blockSize>>>(
        A, B, C1, C2, lda, ldb, ldc1, ldc2, M, N,
        alpha, beta, gamma, delta, out_coef1, out_coef2);
    
    cudaDeviceSynchronize();
}

/**
 * Kernel for computing linear combination of three matrices with accumulation:
 * D = alpha * A + beta * B + gamma * C + out_coef * D
 * 
 * This kernel allows computing a three-way linear combination and accumulating
 * with the existing value in a single operation, avoiding temporary storage.
 * 
 * Parameters:
 * - A: First input matrix (M x N)
 * - B: Second input matrix (M x N)
 * - C: Third input matrix (M x N)
 * - D: Output matrix (M x N)
 * - alpha: Coefficient for matrix A
 * - beta: Coefficient for matrix B
 * - gamma: Coefficient for matrix C
 * - out_coef: Coefficient for accumulation with existing D values
 */
__global__ void sum_3_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B,
    const Float* __restrict__ C, Float* __restrict__ D,
    int lda, int ldb, int ldc, int ldd, int M, int N,
    Float alpha, Float beta, Float gamma, Float out_coef) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Compute linear combination and accumulate in a single operation
        D[row + col * ldd] = alpha * A[row + col * lda] +
                            beta * B[row + col * ldb] +
                            gamma * C[row + col * ldc] +
                            out_coef * D[row + col * ldd];
    }
}

/**
 * Host wrapper for computing linear combination of three matrices with accumulation
 */
void GPU_sum_3(Float *A, Float *B, Float *C, Float *D,
               int lda, int ldb, int ldc, int ldd,
               int M, int N,
               Float alpha, Float beta, Float gamma,
               Float out_coef) {
    
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    sum_3_kernel<<<gridSize, blockSize>>>(
        A, B, C, D, lda, ldb, ldc, ldd, M, N,
        alpha, beta, gamma, out_coef);
    
    cudaDeviceSynchronize();
}

/**
 * Kernel for matrix addition with accumulation:
 * C = alpha * A + beta * B + out_coef * C
 * 
 * This matches the behavior of cublasGeam but adds accumulation capability
 * 
 * Parameters:
 * - A: First input matrix (M x N)
 * - B: Second input matrix (M x N)
 * - C: Output matrix (M x N)
 * - alpha: Coefficient for matrix A
 * - beta: Coefficient for matrix B
 * - out_coef: Coefficient for accumulation with existing C values
 */
__global__ void add_kernel(
    const Float* __restrict__ A, const Float* __restrict__ B, Float* __restrict__ C,
    int lda, int ldb, int ldc, int M, int N,
    Float alpha, Float beta, Float out_coef) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        C[row + col * ldc] = alpha * A[row + col * lda] +
                            beta * B[row + col * ldb] +
                            out_coef * C[row + col * ldc];
    }
}

/**
 * Host wrapper for matrix addition with accumulation
 */
void GPU_add_acc(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int M, int N,
    Float alpha, Float beta,
    Float out_coef) {
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    add_kernel<<<gridSize, blockSize>>>(
        A, B, C, lda, ldb, ldc, M, N,
        alpha, beta, out_coef);
    
    cudaDeviceSynchronize();
}

// Host wrapper functions

/**
 * Host wrapper for D = alpha * A @ (beta * B + gamma * C)^T
 */
void GPU_A_mul_B_plus_C_t(Float *A, Float *B, Float *C, Float *D,
                          int lda, int ldb, int ldc, int ldd,
                          int M, int N, int K,
                          Float alpha, Float beta, Float gamma,
                          Float out_coef) {
    
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    if (M >= 64 && N >= 64 && K >= 64) {
        // Use tiled version for larger matrices
        fused_A_mul_B_plus_C_transpose_tiled_kernel<TILE_DIM><<<gridSize, blockSize>>>(
            A, B, C, D, lda, ldb, ldc, ldd, M, N, K, alpha, beta, gamma, out_coef);
    } else {
        // Use simple version for smaller matrices
        dim3 simpleBlockSize(16, 16);
        dim3 simpleGridSize((N + 15) / 16, (M + 15) / 16);
        fused_A_mul_B_plus_C_transpose_kernel<<<simpleGridSize, simpleBlockSize>>>(
            A, B, C, D, lda, ldb, ldc, ldd, M, N, K, alpha, beta, gamma, out_coef);
    }
    
    cudaDeviceSynchronize();
}

/**
 * Host wrapper for D = alpha * (beta * A + gamma * B) @ C^T
 */
void GPU_A_plus_B_mul_C_t(Float *A, Float *B, Float *C, Float *D,
                          int lda, int ldb, int ldc, int ldd,
                          int M, int N, int K,
                          Float alpha, Float beta, Float gamma,
                          Float out_coef) {
    
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    if (M >= 64 && N >= 64 && K >= 64) {
        // Use tiled version for larger matrices
        fused_A_plus_B_mul_C_transpose_tiled_kernel<TILE_DIM><<<gridSize, blockSize>>>(
            A, B, C, D, lda, ldb, ldc, ldd, M, N, K, alpha, beta, gamma, out_coef);
    } else {
        // Use simple version for smaller matrices
        dim3 simpleBlockSize(16, 16);
        dim3 simpleGridSize((N + 15) / 16, (M + 15) / 16);
        fused_A_plus_B_mul_C_transpose_kernel<<<simpleGridSize, simpleBlockSize>>>(
            A, B, C, D, lda, ldb, ldc, ldd, M, N, K, alpha, beta, gamma, out_coef);
    }
    
    cudaDeviceSynchronize();
} 