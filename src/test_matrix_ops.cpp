#include <cstdio>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "strassen.cpp"

// Helper function to print matrix
void print_matrix(const char* name, Float* mat, int rows, int cols, int ld) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i + j * ld]);
        }
        printf("\n");
    }
}

// Helper function to initialize matrix with pattern
void init_matrix(Float* mat, int rows, int cols, int ld, int start = 1) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            mat[i + j * ld] = start + i + j * rows;
        }
    }
}

// Test 4x4 matrix operations
void test_4x4() {
    printf("\n=== Testing 4x4 Matrix Operations ===\n");
    
    // Allocate host memory
    Float *h_A = (Float*)malloc(4 * 4 * sizeof(Float));
    Float *h_C = (Float*)malloc(4 * 4 * sizeof(Float));
    
    // Initialize matrices
    init_matrix(h_A, 4, 4, 4);
    for (int i = 0; i < 16; i++) h_C[i] = 0;
    
    // Allocate device memory
    Float *d_A, *d_C;
    cudaMalloc((void**)&d_A, 4 * 4 * sizeof(Float));
    cudaMalloc((void**)&d_C, 4 * 4 * sizeof(Float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, 4 * 4 * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, 4 * 4 * sizeof(Float), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Test GPU_T
    printf("\nTesting GPU_T:\n");
    GPU_T(d_A, d_C, 4, 4, 4, 4);
    cudaMemcpy(h_C, d_C, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("Original A", h_A, 4, 4, 4);
    print_matrix("Transposed A", h_C, 4, 4, 4);
    
    // Test GPU_ABt
    printf("\nTesting GPU_ABt:\n");
    cudaMemset(d_C, 0, 4 * 4 * sizeof(Float));
    GPU_ABt(d_A, d_A, d_C, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1.0, 0.0);
    cudaMemcpy(h_C, d_C, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("A * A^T", h_C, 4, 4, 4);
    
    // Test GPU_AtB
    printf("\nTesting GPU_AtB:\n");
    cudaMemset(d_C, 0, 4 * 4 * sizeof(Float));
    GPU_AtB(d_A, d_A, d_C, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1.0, 0.0);
    cudaMemcpy(h_C, d_C, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("A^T * A", h_C, 4, 4, 4);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
}

// Test 16x16 matrix operations with 16-part split
void test_16x16() {
    printf("\n=== Testing 16x16 Matrix Operations with 16-part Split ===\n");
    
    const int N = 16;
    const int N4 = N/4;
    
    // Allocate host memory
    Float *h_A = (Float*)malloc(N * N * sizeof(Float));
    Float *h_C = (Float*)malloc(N * N * sizeof(Float));
    
    // Initialize matrices
    init_matrix(h_A, N, N, N);
    for (int i = 0; i < N*N; i++) h_C[i] = 0;
    
    // Allocate device memory
    Float *d_A, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(Float));
    cudaMalloc((void**)&d_C, N * N * sizeof(Float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Create matrix views for input matrix A
    Matrix A_mat(d_A, N, N, N);
    Matrix X1 = A_mat.view(     0,    0,  N4, N4);
    Matrix X2 = A_mat.view(   N4,    0,  N4, N4);
    Matrix X3 = A_mat.view( 2*N4,    0,  N4, N4);
    Matrix X4 = A_mat.view( 3*N4,    0,  N4, N4);
    Matrix X5 = A_mat.view(     0,   N4,  N4, N4);
    Matrix X6 = A_mat.view(   N4,   N4,  N4, N4);
    Matrix X7 = A_mat.view( 2*N4,   N4,  N4, N4);
    Matrix X8 = A_mat.view( 3*N4,   N4,  N4, N4);
    Matrix X9 = A_mat.view(     0, 2*N4,  N4, N4);
    Matrix X10 = A_mat.view(  N4, 2*N4,  N4, N4);
    Matrix X11 = A_mat.view(2*N4, 2*N4,  N4, N4);
    Matrix X12 = A_mat.view(3*N4, 2*N4,  N4, N4);
    Matrix X13 = A_mat.view(    0, 3*N4,  N4, N4);
    Matrix X14 = A_mat.view(  N4, 3*N4,  N4, N4);
    Matrix X15 = A_mat.view(2*N4, 3*N4,  N4, N4);
    Matrix X16 = A_mat.view(3*N4, 3*N4,  N4, N4);
    
    // Create matrix views for output matrix C
    Matrix C_mat(d_C, N, N, N);
    Matrix C11 = C_mat.view(    0,     0, N4, N4);
    Matrix C12 = C_mat.view(  N4,     0, N4, N4);
    Matrix C13 = C_mat.view(2*N4,     0, N4, N4);
    Matrix C14 = C_mat.view(3*N4,     0, N4, N4);
    Matrix C21 = C_mat.view(    0,   N4, N4, N4);
    Matrix C22 = C_mat.view(  N4,   N4, N4, N4);
    Matrix C23 = C_mat.view(2*N4,   N4, N4, N4);
    Matrix C24 = C_mat.view(3*N4,   N4, N4, N4);
    Matrix C31 = C_mat.view(    0, 2*N4, N4, N4);
    Matrix C32 = C_mat.view(  N4, 2*N4, N4, N4);
    Matrix C33 = C_mat.view(2*N4, 2*N4, N4, N4);
    Matrix C34 = C_mat.view(3*N4, 2*N4, N4, N4);
    Matrix C41 = C_mat.view(    0, 3*N4, N4, N4);
    Matrix C42 = C_mat.view(  N4, 3*N4, N4, N4);
    Matrix C43 = C_mat.view(2*N4, 3*N4, N4, N4);
    Matrix C44 = C_mat.view(3*N4, 3*N4, N4, N4);
    
    // Test GPU_T on submatrices
    printf("\nTesting GPU_T on submatrices:\n");
    GPU_T(X1, C11);
    GPU_T(X2, C12);
    GPU_T(X3, C13);
    GPU_T(X4, C14);
    
    cudaMemcpy(h_C, d_C, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("Transposed submatrices", h_C, N, N, N);
    
    // Test GPU_ABt on submatrices
    printf("\nTesting GPU_ABt on submatrices:\n");
    cudaMemset(d_C, 0, N * N * sizeof(Float));
    GPU_ABt(X1, X2, C11);
    GPU_ABt(X3, X4, C12);
    GPU_ABt(X5, X6, C21);
    GPU_ABt(X7, X8, C22);
    
    cudaMemcpy(h_C, d_C, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("Submatrix multiplications", h_C, N, N, N);
    
    // Test GPU_AtB on submatrices
    printf("\nTesting GPU_AtB on submatrices:\n");
    cudaMemset(d_C, 0, N * N * sizeof(Float));
    GPU_AtB(X1, X5, C11);
    GPU_AtB(X2, X6, C12);
    GPU_AtB(X3, X7, C21);
    GPU_AtB(X4, X8, C22);
    
    cudaMemcpy(h_C, d_C, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("Submatrix transposed multiplications", h_C, N, N, N);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
}

int main() {
    test_4x4();
    test_16x16();
    return 0;
} 