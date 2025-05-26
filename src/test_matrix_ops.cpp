#include <cstdio>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "rtxx.cpp"

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

// Helper function to compare matrices
bool compare_matrices(Float* mat1, Float* mat2, int rows, int cols, int ld1, int ld2) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if (std::abs(mat1[i + j * ld1] - mat2[i + j * ld2]) > 1e-6) {
                printf("Mismatch at (%d,%d): %.6f vs %.6f\n", i, j, 
                       mat1[i + j * ld1], mat2[i + j * ld2]);
                return false;
            }
        }
    }
    return true;
}

// Helper function to initialize matrix with pattern
void init_matrix(Float* mat, int rows, int cols, int ld, int start = 1) {
    // Initialize the full allocated memory
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < ld; i++) {
            if (i < rows) {
                mat[i + j * ld] = start + i + j * rows;
            } else {
                mat[i + j * ld] = 0;  // Initialize padding to zero
            }
        }
    }
}

// Test 4x4 matrix operations
void test_4x4() {
    printf("\n=== Testing 4x4 Matrix Operations ===\n");
    
    // Allocate host memory
    Float *h_A = (Float*)malloc(4 * 4 * sizeof(Float));
    Float *h_C1 = (Float*)malloc(4 * 4 * sizeof(Float));  // For Matrix objects
    Float *h_C2 = (Float*)malloc(4 * 4 * sizeof(Float));  // For raw pointers
    
    // Initialize matrices
    init_matrix(h_A, 4, 4, 4);
    h_C1[0] = 77;
    h_C2[0] = 77;
    h_C1[1] = 3;
    h_C2[1] = 3;
    for (int i = 2; i < 16; i++) {
        h_C1[i] = -1;
        h_C2[i] = -1;
    }
    
    // Allocate device memory
    Float *d_A, *d_C1, *d_C2;
    cudaMalloc((void**)&d_A, 4 * 4 * sizeof(Float));
    cudaMalloc((void**)&d_C1, 4 * 4 * sizeof(Float));
    cudaMalloc((void**)&d_C2, 4 * 4 * sizeof(Float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, 4 * 4 * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, h_C1, 4 * 4 * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2, h_C2, 4 * 4 * sizeof(Float), cudaMemcpyHostToDevice);
    
    // Create Matrix objects
    Matrix A(d_A, 4, 4, 4);
    Matrix C1(d_C1, 4, 4, 4);
    Matrix C2(d_C2, 4, 4, 4);
    
    // // Test GPU_add
    printf("\nTesting GPU_add:\n");
    // Using Matrix objects
    GPU_add(A, A, C1);
    // Using raw pointers
    GPU_add(d_A, d_A, d_C2, 4, 4, 4, 4, 4, 1.0, 1.0);
    
    cudaMemcpy(h_C1, d_C1, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("Original A", h_A, 4, 4, 4);
    print_matrix("A + A (Matrix objects)", h_C1, 4, 4, 4);
    print_matrix("A + A (raw pointers)", h_C2, 4, 4, 4);
    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, 4, 4, 4, 4) ? "Yes" : "No");

    // Test GPU_sub
    printf("\nTesting GPU_sub:\n");
    cudaMemset(d_C1, 0, 4 * 4 * sizeof(Float));
    cudaMemset(d_C2, 0, 4 * 4 * sizeof(Float));
    
    // Using Matrix objects
    GPU_sub(A, A, C1);
    // Using raw pointers
    GPU_add(d_A, d_A, d_C2, 4, 4, 4, 4, 4, 1.0, -1.0);
    
    cudaMemcpy(h_C1, d_C1, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("A - A (Matrix objects)", h_C1, 4, 4, 4);
    print_matrix("A - A (raw pointers)", h_C2, 4, 4, 4);
    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, 4, 4, 4, 4) ? "Yes" : "No");
    
    // Test GPU_T
    printf("\nTesting GPU_T:\n");

    printf("Output matrix before transpose:\n");
    print_matrix("C1", h_C1, 4, 4, 4);
    // Using Matrix objects
    GPU_T(A, C1);
    // Using raw pointers
    GPU_T(d_A, d_C2, 4, 4, 4, 4);
    
    cudaMemcpy(h_C1, d_C1, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("Original A", h_A, 4, 4, 4);
    print_matrix("Transposed A (Matrix objects)", h_C1, 4, 4, 4);
    print_matrix("Transposed A (raw pointers)", h_C2, 4, 4, 4);
    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, 4, 4, 4, 4) ? "Yes" : "No");
    
    // Test GPU_ABt
    printf("\nTesting GPU_ABt:\n");
    cudaMemset(d_C1, 0, 4 * 4 * sizeof(Float));
    cudaMemset(d_C2, 0, 4 * 4 * sizeof(Float));
    
    // Using Matrix objects
    GPU_ABt(A, A, C1);
    // Using raw pointers
    GPU_ABt(d_A, d_A, d_C2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1.0, 0.0);
    
    cudaMemcpy(h_C1, d_C1, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("A * A^T (Matrix objects)", h_C1, 4, 4, 4);
    print_matrix("A * A^T (raw pointers)", h_C2, 4, 4, 4);
    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, 4, 4, 4, 4) ? "Yes" : "No");
    
    // Test GPU_AtB
    printf("\nTesting GPU_AtB:\n");
    cudaMemset(d_C1, 0, 4 * 4 * sizeof(Float));
    cudaMemset(d_C2, 0, 4 * 4 * sizeof(Float));
    
    // Using Matrix objects
    GPU_AtB(A, A, C1);
    // Using raw pointers
    GPU_AtB(d_A, d_A, d_C2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1.0, 0.0);
    
    cudaMemcpy(h_C1, d_C1, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("Original A", h_A, 4, 4, 4);
    print_matrix("A^T * A (Matrix objects)", h_C1, 4, 4, 4);
    print_matrix("A^T * A (raw pointers)", h_C2, 4, 4, 4);
    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, 4, 4, 4, 4) ? "Yes" : "No");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C1);
    cudaFree(d_C2);
    free(h_A);
    free(h_C1);
    free(h_C2);
}

// Test 16x16 matrix operations with 16-part split
void test_16x16() {
    printf("\n=== Testing 16x16 Matrix Operations with 16-part Split ===\n");
    
    const int N = 16;
    const int N4 = N/4;
    
    // Allocate host memory
    Float *h_A = (Float*)malloc(N * N * sizeof(Float));
    Float *h_C1 = (Float*)malloc(N * N * sizeof(Float));  // For Matrix objects
    Float *h_C2 = (Float*)malloc(N * N * sizeof(Float));  // For raw pointers
    
    // Initialize matrices
    init_matrix(h_A, N, N, N);
    for (int i = 0; i < N*N; i++) {
        h_C1[i] = 0;
        h_C2[i] = 0;
    }
    
    // Allocate device memory
    Float *d_A, *d_C1, *d_C2;
    cudaMalloc((void**)&d_A, N * N * sizeof(Float));
    cudaMalloc((void**)&d_C1, N * N * sizeof(Float));
    cudaMalloc((void**)&d_C2, N * N * sizeof(Float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, h_C1, N * N * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2, h_C2, N * N * sizeof(Float), cudaMemcpyHostToDevice);

    Float *C11_p = h_C1;
    Float *C12_p = h_C1 + N4;
    Float *C13_p = h_C1 + 2*N4;
    Float *C14_p = h_C1 + 3*N4;
    Float *C21_p = h_C1 + N*N4;
    Float *C22_p = h_C1 + N4 + N*N4;
    Float *C23_p = h_C1 + 2*N4 + N*N4;
    Float *C24_p = h_C1 + 3*N4 + N*N4;
    Float *C31_p = h_C1 + 2*N*N4;
    Float *C32_p = h_C1 + 3*N*N4;
    Float *C33_p = h_C1 + 2*N4 + 2*N*N4;
    Float *C34_p = h_C1 + 3*N4 + 2*N*N4;
    Float *C41_p = h_C1 + 3*N*N4;
    Float *C42_p = h_C1 + 3*N4 + 3*N*N4;
    Float *C43_p = h_C1 + 2*N4 + 3*N*N4;
    Float *C44_p = h_C1 + 3*N4 + 3*N*N4;
    
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
    Matrix C1_mat(d_C1, N, N, N);
    Matrix C2_mat(d_C2, N, N, N);
    Matrix C11 = C1_mat.view(    0,     0, N4, N4);
    Matrix C12 = C1_mat.view(  N4,     0, N4, N4);
    Matrix C13 = C1_mat.view(2*N4,     0, N4, N4);
    Matrix C14 = C1_mat.view(3*N4,     0, N4, N4);
    Matrix C21 = C1_mat.view(    0,   N4, N4, N4);
    Matrix C22 = C1_mat.view(  N4,   N4, N4, N4);
    Matrix C23 = C1_mat.view(2*N4,   N4, N4, N4);
    Matrix C24 = C1_mat.view(3*N4,   N4, N4, N4);
    Matrix C31 = C1_mat.view(    0, 2*N4, N4, N4);
    Matrix C32 = C1_mat.view(  N4, 2*N4, N4, N4);
    Matrix C33 = C1_mat.view(2*N4, 2*N4, N4, N4);
    Matrix C34 = C1_mat.view(3*N4, 2*N4, N4, N4);
    Matrix C41 = C1_mat.view(    0, 3*N4, N4, N4);
    Matrix C42 = C1_mat.view(  N4, 3*N4, N4, N4);
    Matrix C43 = C1_mat.view(2*N4, 3*N4, N4, N4);
    Matrix C44 = C1_mat.view(3*N4, 3*N4, N4, N4);

    // Test GPU_T on submatrices
    printf("\nTesting GPU_T on submatrices:\n");

    printf("Input matrix:");
    print_matrix("A", h_A, N, N, N);
    // Using Matrix objects
    GPU_T(X1, C11);
    GPU_T(X2, C12);
    GPU_T(X3, C13);
    GPU_T(X4, C14);
    GPU_T(X11, C33);

    cudaMemcpy(h_C1, d_C1, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("Transposed submatrices (Matrix objects)", h_C1, N, N, N);

    printf("Results match: %s\n", compare_matrices(h_C1, h_C2, N, N, N, N) ? "Yes" : "No");
    
    // Test GPU_ABt on submatrices
    printf("\nTesting GPU_ABt on submatrices:\n");
    cudaMemset(d_C1, 0, N * N * sizeof(Float));
    
    // Using Matrix objects
    GPU_ABt(X1, X2, C11);
    GPU_ABt(X3, X4, C12);
    GPU_ABt(X2, X5, C22);
    GPU_ABt(X5, X6, C34);
    GPU_ABt(X7, X8, C44);
    
    cudaMemcpy(h_C1, d_C1, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("Submatrix multiplications (Matrix objects)", h_C1, N, N, N);
    
    // Test GPU_AtB on submatrices
    printf("\nTesting GPU_AtB on submatrices:\n");
    cudaMemset(d_C1, 0, N * N * sizeof(Float));
    
    // Using Matrix objects
    GPU_AtB(X1, X5, C11);
    GPU_AtB(X4, X8, C22);
    GPU_AtB(X2, X6, C41);
    
    cudaMemcpy(h_C1, d_C1, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    
    print_matrix("Submatrix transposed multiplications (Matrix objects)", h_C1, N, N, N);

    printf("Testing reusing C:");
    cudaMemset(d_C1, 0, N * N * sizeof(Float));

    GPU_add(X1, X2, C11);
    GPU_add(X1, X2, C12);
    GPU_add(C12, C11, C22);
    GPU_ABt(C11, C12, C33);
    GPU_AtB(C11, C12, C44);

    cudaMemcpy(h_C1, d_C1, N * N * sizeof(Float), cudaMemcpyDeviceToHost);
    print_matrix("C matrix after several operations", h_C1, N, N, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C1);
    cudaFree(d_C2);
    free(h_A);
    free(h_C1);
    free(h_C2);
}

int main() {
    // Initialize cuBLAS handle
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cuBLAS initialization error\n");
        return EXIT_FAILURE;
    }
    
    test_4x4();
    test_16x16();
    
    // Cleanup cuBLAS handle
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cuBLAS shutdown error\n");
        return EXIT_FAILURE;
    }
    
    return 0;
} 