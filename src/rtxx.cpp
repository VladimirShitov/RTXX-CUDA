#include <cstdio>
#include <cuda_runtime_api.h>

#include "strassen.cpp"

cublasHandle_t handle;

struct Matrix {
    Float* data;      // Pointer to matrix data
    int rows;         // Number of rows
    int cols;         // Number of columns
    int ld;           // Leading dimension
    
    // Constructor for view (no memory ownership)
    Matrix(Float* d, int r, int c, int l) 
        : data(d), rows(r), cols(c), ld(l) {}
    
    // Default constructor
    Matrix() : data(nullptr), rows(0), cols(0), ld(0) {}
    
    // Create a view of a submatrix
    Matrix view(int start_row, int start_col, int num_rows, int num_cols) {
        return Matrix(data + start_row + start_col * ld, num_rows, num_cols, ld);
    }
};

// Matrix operations
void GPU_add(const Matrix& A, const Matrix& B, Matrix& C, Float alpha = 1.0, Float beta = 1.0) {
    GPU_add(A.data, B.data, C.data, A.ld, B.ld, C.ld, A.rows, A.cols, alpha, beta);
}

void GPU_sub(const Matrix& A, const Matrix& B, Matrix& C, Float alpha = 1.0, Float beta = -1.0) {
    GPU_add(A.data, B.data, C.data, A.ld, B.ld, C.ld, A.rows, A.cols, alpha, beta);
}

// Original GPU_ABt function for raw pointers
void GPU_ABt(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta) {
    // C = alpha * (A * B^T) + beta * C
    // A is XA x YA
    // B is XB x YB
    // C is XA x XB
    cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, XA, XB, YA, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// New overload for Matrix objects
void GPU_ABt(const Matrix& A, const Matrix& B, Matrix& C, Float alpha = 1.0, Float beta = 0.0) {
    GPU_ABt(A.data, B.data, C.data, A.ld, B.ld, C.ld, 
            A.rows, B.rows, C.rows, A.cols, B.cols, C.cols, alpha, beta);
}

// Original GPU_T function for raw pointers
void GPU_T(Float *A, Float *C, int lda, int ldc, int XA, int YA, Float alpha = 1.0) {
    Float zero = 0.0;
    cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &zero, C, ldc, C, ldc);
}

// Matrix overload for GPU_T
void GPU_T(const Matrix& A, Matrix& C, Float alpha = 1.0) {
    GPU_T(A.data, C.data, A.ld, C.ld, A.rows, A.cols, alpha);
}

void GPU_AtB_strassen(const Matrix& A, const Matrix& B, Matrix& C, Matrix& A2t, int depth, Float alpha = 1.0) {
    GPU_T(A, A2t, alpha);
    strassen(A2t.data, B.data, C.data, A2t.ld, B.ld, C.ld, 
             A2t.rows, B.rows, C.rows, A2t.cols, B.cols, C.cols, depth - 1);
}

void GPU_ABt_strassen(const Matrix& A, const Matrix& B, Matrix& C, Matrix& Xt, int depth, Float alpha = 1.0) {
    GPU_T(B, Xt, alpha);
    strassen(A.data, Xt.data, C.data, A.ld, Xt.ld, C.ld, 
             A.rows, Xt.rows, C.rows, A.cols, Xt.cols, C.cols, depth);
}

// Original GPU_AtB function for raw pointers
void GPU_AtB(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta) {
    cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

// New overload for Matrix objects
void GPU_AtB(const Matrix& A, const Matrix& B, Matrix& C, Float alpha = 1.0, Float beta = 0.0) {
    GPU_AtB(A.data, B.data, C.data, A.ld, B.ld, C.ld, 
            A.rows, B.rows, C.rows, A.cols, B.cols, C.cols, alpha, beta);
}

// Helper function to print 4x4 elements of a matrix
void print_matrix_4x4(const char* name, const Matrix& mat) {
    printf("\n%s (first 4x4 elements):\n", name);
    Float* host_data = new Float[4 * 4];
    cudaMemcpy(host_data, mat.data, 4 * 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%8.2f ", host_data[i * mat.ld + j]);
        }
        printf("\n");
    }
    delete[] host_data;
}

void rtxx(Float *A, Float *C, int lda, int ldc,
          int XA, int XC, int YA, int YC, int depth) {
    int XA4 = XA / 4;
    int XC4 = XC / 4;
    int YA4 = YA / 4;
    int YC4 = YC / 4;

    Float *W_1, *W_2;
    int ldw = XC4;
    int ldm = XC4;

    // Allocate memory with proper error checking
    cudaError_t err;
    err = cudaMalloc((void **)&W_1, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { 
        printf("Failed to allocate W_1: %s\n", cudaGetErrorString(err));
        printf("Attempted to allocate %lu bytes\n", (size_t)ldw * YC4 * sizeof(Float));
        return; 
    }
    err = cudaMalloc((void **)&W_2, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate W_2: %s\n", cudaGetErrorString(err)); return; }

    int dXA = XA4;
    int dYA = YA4 * lda;
    int dXC = XC4;
    int dYC = YC4 * ldc;

    // Create matrix views for input matrix A
    Matrix A_mat(A, XA, YA, lda);

    // Normal version
    // Matrix X1 = A_mat.view(     0,      0, XA4, YA4);
    // Matrix X2 = A_mat.view(   XA4,      0, XA4, YA4);
    // Matrix X3 = A_mat.view(2*XA4,       0, XA4, YA4);
    // Matrix X4 = A_mat.view(3*XA4,       0, XA4, YA4);
    // Matrix X5 = A_mat.view(     0,    YA4, XA4, YA4);
    // Matrix X6 = A_mat.view(   XA4,    YA4, XA4, YA4);
    // Matrix X7 = A_mat.view( 2*XA4,    YA4, XA4, YA4);
    // Matrix X8 = A_mat.view( 3*XA4,    YA4, XA4, YA4);
    // Matrix X9 = A_mat.view(     0,  2*YA4, XA4, YA4);
    // Matrix X10 = A_mat.view(  XA4,  2*YA4, XA4, YA4);
    // Matrix X11 = A_mat.view(2*XA4,  2*YA4, XA4, YA4);
    // Matrix X12 = A_mat.view(3*XA4,  2*YA4, XA4, YA4);
    // Matrix X13 = A_mat.view(    0,  3*YA4, XA4, YA4);
    // Matrix X14 = A_mat.view(  XA4,  3*YA4, XA4, YA4);
    // Matrix X15 = A_mat.view(2*XA4,  3*YA4, XA4, YA4);
    // Matrix X16 = A_mat.view(3*XA4,  3*YA4, XA4, YA4);

    // Transposed version
    Matrix X1 = A_mat.view(     0,      0, XA4, YA4);
    Matrix X2 = A_mat.view(     0,    YA4, XA4, YA4);
    Matrix X3 = A_mat.view(     0,  2*YA4, XA4, YA4);
    Matrix X4 = A_mat.view(     0,  3*YA4, XA4, YA4);
    Matrix X5 = A_mat.view(   XA4,     0, XA4, YA4);
    Matrix X6 = A_mat.view(   XA4,    YA4, XA4, YA4);
    Matrix X7 = A_mat.view(   XA4,  2*YA4, XA4, YA4);
    Matrix X8 = A_mat.view(   XA4,  3*YA4, XA4, YA4);
    Matrix X9 = A_mat.view( 2*XA4,      0, XA4, YA4);
    Matrix X10 = A_mat.view(2*XA4,    YA4, XA4, YA4);
    Matrix X11 = A_mat.view(2*XA4,  2*YA4, XA4, YA4);
    Matrix X12 = A_mat.view(2*XA4,  3*YA4, XA4, YA4);
    Matrix X13 = A_mat.view(3*XA4,      0, XA4, YA4);
    Matrix X14 = A_mat.view(3*XA4,    YA4, XA4, YA4);
    Matrix X15 = A_mat.view(3*XA4,  2*YA4, XA4, YA4);
    Matrix X16 = A_mat.view(3*XA4,  3*YA4, XA4, YA4);

    // Create matrix views for output matrix C
    Matrix C_mat(C, XC, YC, ldc);
    // Matrix C11 = C_mat.view(    0,     0, XC4, YC4);
    // Matrix C12 = C_mat.view(  XC4,     0, XC4, YC4);
    // Matrix C13 = C_mat.view(2*XC4,     0, XC4, YC4);
    // Matrix C14 = C_mat.view(3*XC4,     0, XC4, YC4);
    // Matrix C21 = C_mat.view(    0,   YC4, XC4, YC4);
    // Matrix C22 = C_mat.view(  XC4,   YC4, XC4, YC4);
    // Matrix C23 = C_mat.view(2*XC4,   YC4, XC4, YC4);
    // Matrix C24 = C_mat.view(3*XC4,   YC4, XC4, YC4);
    // Matrix C31 = C_mat.view(    0, 2*YC4, XC4, YC4);
    // Matrix C32 = C_mat.view(  XC4, 2*YC4, XC4, YC4);
    // Matrix C33 = C_mat.view(2*XC4, 2*YC4, XC4, YC4);
    // Matrix C34 = C_mat.view(3*XC4, 2*YC4, XC4, YC4);
    // Matrix C41 = C_mat.view(    0, 3*YC4, XC4, YC4);
    // Matrix C42 = C_mat.view(  XC4, 3*YC4, XC4, YC4);
    // Matrix C43 = C_mat.view(2*XC4, 3*YC4, XC4, YC4);
    // Matrix C44 = C_mat.view(3*XC4, 3*YC4, XC4, YC4);

    // Transposed version
    Matrix C11 = C_mat.view(    0,     0    , XC4, YC4);
    Matrix C12 = C_mat.view(    0,     YA4  , XC4, YC4);
    Matrix C13 = C_mat.view(    0,     2*YA4, XC4, YC4);
    Matrix C14 = C_mat.view(    0,     3*YA4, XC4, YC4);
    Matrix C21 = C_mat.view(  XC4,     0    , XC4, YC4);
    Matrix C22 = C_mat.view(  XC4,     YA4  , XC4, YC4);
    Matrix C23 = C_mat.view  (XC4,     2*YA4, XC4, YC4);
    Matrix C24 = C_mat.view  (XC4,     3*YA4, XC4, YC4);
    Matrix C31 = C_mat.view(2*XC4,     0    , XC4, YC4);
    Matrix C32 = C_mat.view(2*XC4,     YA4  , XC4, YC4);
    Matrix C33 = C_mat.view(2*XC4,     2*YA4, XC4, YC4);
    Matrix C34 = C_mat.view(2*XC4,     3*YA4, XC4, YC4);
    Matrix C41 = C_mat.view(3*XC4,     0    , XC4, YC4);
    Matrix C42 = C_mat.view(3*XC4,     YA4  , XC4, YC4);
    Matrix C43 = C_mat.view(3*XC4,     2*YA4, XC4, YC4);
    Matrix C44 = C_mat.view(3*XC4,     3*YA4, XC4, YC4);    

    // Debug prints for matrix views
    printf("\nInput matrix A views:\n");
    printf("X1:  pos=(%d,%d) size=%dx%d ld=%d\n", 0, 0, X1.rows, X1.cols, X1.ld);
    printf("X2:  pos=(%d,%d) size=%dx%d ld=%d\n", dXA, 0, X2.rows, X2.cols, X2.ld);
    printf("X3:  pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXA, 0, X3.rows, X3.cols, X3.ld);
    printf("X4:  pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXA, 0, X4.rows, X4.cols, X4.ld);
    printf("X5:  pos=(%d,%d) size=%dx%d ld=%d\n", 0, dYA, X5.rows, X5.cols, X5.ld);
    printf("X6:  pos=(%d,%d) size=%dx%d ld=%d\n", dXA, dYA, X6.rows, X6.cols, X6.ld);
    printf("X7:  pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXA, dYA, X7.rows, X7.cols, X7.ld);
    printf("X8:  pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXA, dYA, X8.rows, X8.cols, X8.ld);
    printf("X9:  pos=(%d,%d) size=%dx%d ld=%d\n", 0, 2*dYA, X9.rows, X9.cols, X9.ld);
    printf("X10: pos=(%d,%d) size=%dx%d ld=%d\n", dXA, 2*dYA, X10.rows, X10.cols, X10.ld);
    printf("X11: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXA, 2*dYA, X11.rows, X11.cols, X11.ld);
    printf("X12: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXA, 2*dYA, X12.rows, X12.cols, X12.ld);
    printf("X13: pos=(%d,%d) size=%dx%d ld=%d\n", 0, 3*dYA, X13.rows, X13.cols, X13.ld);
    printf("X14: pos=(%d,%d) size=%dx%d ld=%d\n", dXA, 3*dYA, X14.rows, X14.cols, X14.ld);
    printf("X15: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXA, 3*dYA, X15.rows, X15.cols, X15.ld);
    printf("X16: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXA, 3*dYA, X16.rows, X16.cols, X16.ld);

    printf("\nOutput matrix C views:\n");
    printf("C11: pos=(%d,%d) size=%dx%d ld=%d\n", 0, 0, C11.rows, C11.cols, C11.ld);
    printf("C12: pos=(%d,%d) size=%dx%d ld=%d\n", dXC, 0, C12.rows, C12.cols, C12.ld);
    printf("C13: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXC, 0, C13.rows, C13.cols, C13.ld);
    printf("C14: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXC, 0, C14.rows, C14.cols, C14.ld);
    printf("C21: pos=(%d,%d) size=%dx%d ld=%d\n", 0, dYC, C21.rows, C21.cols, C21.ld);
    printf("C22: pos=(%d,%d) size=%dx%d ld=%d\n", dXC, dYC, C22.rows, C22.cols, C22.ld);
    printf("C23: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXC, dYC, C23.rows, C23.cols, C23.ld);
    printf("C24: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXC, dYC, C24.rows, C24.cols, C24.ld);
    printf("C31: pos=(%d,%d) size=%dx%d ld=%d\n", 0, 2*dYC, C31.rows, C31.cols, C31.ld);
    printf("C32: pos=(%d,%d) size=%dx%d ld=%d\n", dXC, 2*dYC, C32.rows, C32.cols, C32.ld);
    printf("C33: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXC, 2*dYC, C33.rows, C33.cols, C33.ld);
    printf("C34: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXC, 2*dYC, C34.rows, C34.cols, C34.ld);
    printf("C41: pos=(%d,%d) size=%dx%d ld=%d\n", 0, 3*dYC, C41.rows, C41.cols, C41.ld);
    printf("C42: pos=(%d,%d) size=%dx%d ld=%d\n", dXC, 3*dYC, C42.rows, C42.cols, C42.ld);
    printf("C43: pos=(%d,%d) size=%dx%d ld=%d\n", 2*dXC, 3*dYC, C43.rows, C43.cols, C43.ld);
    printf("C44: pos=(%d,%d) size=%dx%d ld=%d\n", 3*dXC, 3*dYC, C44.rows, C44.cols, C44.ld);
    printf("\n");

    // Create matrix views for temporary matrices
    Matrix W1_mat(W_1, XC4, YC4, ldw);
    Matrix W2_mat(W_2, XC4, YC4, ldw);

    /* cutoff criteria */
    float mm = (float)CUTOFF / XA4;
    float nn = (float)CUTOFF / YA4;
    bool stop = (mm + nn) >= 2;

    if (depth <= 1 || stop) {
        // Fill in lower left triangle of the matrix
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                Matrix C_curr = C_mat.view(i*XC4, j*YC4, XC4, YC4);

                for(int k=0; k<4; k++) {
                    Matrix X_i = A_mat.view(i*XA4, k*YA4, XA4, YA4);
                    Matrix X_j = A_mat.view(j*XA4, k*YA4, XA4, YA4);

                    if (k == 0) {
                        GPU_ABt(X_i, X_j, W1_mat, 1.0, 0.0);
                    }
                    else if(k == 1) {
                        GPU_ABt(X_i, X_j, W2_mat, 1.0, 0.0);
                        GPU_add(W1_mat, W2_mat, C_curr);
                    } 
                    else {
                        GPU_ABt(X_i, X_j, W1_mat, 1.0, 0.0);
                        GPU_add(C_curr, W1_mat, C_curr);
                    }  
                }
            }
        }
    }
    else {
        // Apply RTXX with recursive calls for corner matrices
        Float *Xt;
        int ldt = YA4;
        cudaMalloc((void **)&Xt, ldt * XA4 * sizeof(Float));
        Matrix Xt_mat(Xt, YA4, XA4, ldt);

        // TODO: remove hotfix, now the matrix is transformed for some reason
        // GPU_T(A_mat, A_mat);

        print_matrix_4x4("Imput matrix:", A_mat);

        // Memory layout of the matrices:
        // | C11       | C12       | C13       | C14       |
        // | C21       | C22       | C23       | C24       |
        // | C31       | C32       | C33       | C34       |
        // | C41       | C42       | C43       | C44       |

        // y2 = X12 - X10 -> C11
        GPU_sub(X12, X10, C11);
        // | y2        |           |           |           |
        // |           |           |           |           |
        // |           |           |           |           |
        // |           |           |           |           |

        print_matrix_4x4("y2", C11);

        // m17 = X12 * (-y2)^T -> C33
        GPU_ABt(X12, C11, C33, -1.0, 0.0);
        // |           |           |           |           |
        // |           |           |           |           |
        // |           |           | m17       |           |
        // |           |           |           |           |
        // y2 is not needed anymore

        print_matrix_4x4("m17", C11);

        // w5 = X16 + y2 -> C23
        GPU_add(X16, C11, C23);
        // |           |           |           |           |
        // |           |           | w5        |           |
        // |           |           | m17       |           |
        // |           |           |           |           |

        print_matrix_4x4("w5", C11);

        // m3 = (-X2 + X12) * w5^T -> C24
        GPU_sub(X12, X2, W1_mat);
        GPU_ABt(W1_mat, C23, C24, 1.0, 0.0);
        // |           |           |           |           |
        // |           |           | w5        | m3        |
        // |           |           | m17       |           |
        // |           |           |           |           |

        print_matrix_4x4("m3", C11);

        // m24 = (-X1 + X4 + X12) * X16^T -> C13
        GPU_sub(X4, X1, W1_mat);
        GPU_add(W1_mat, X12, W1_mat);
        GPU_ABt(W1_mat, X16, C13, 1.0, 0.0);
        // |           |           | m24       |           |
        // |           |           | w5        | m3        |
        // |           |           | m17       |           |
        // |           |           |           |           |

        print_matrix_4x4("m24", C11);

        // z3 = m3 + m17 - m24 -> C13
        GPU_sub(C33, C13, C13);  // m17 - m24
        GPU_add(C13, C24, C13); 
        // |           |           | z3          |           |
        // |           |           | w5          | m3        |
        // |           |           | m17         |           |
        // |           |           |             |           |

        print_matrix_4x4("z3", C11);
        
        // w3 = X6 + X7 -> C11
        // w4 = X14 + X15 -> C42
        GPU_add(X6, X7, C11);
        GPU_add(X14, X15, C42);
        // | w3        |           | z3          |           |
        // |           |           | w5          | m3        |
        // |           |           | m17         |           |
        // |           | w4        |             |           |

        print_matrix_4x4("w3, w4", C11);
        
        // m8 = X2 * (w3 - w4 + w5)^T -> C34
        GPU_sub(C11, C42, W1_mat);
        GPU_add(W1_mat, C23, W1_mat);
        GPU_ABt(X2, W1_mat, C34, 1.0, 0.0);
        // | w3        |           | z3          |           |
        // |           |           | w5          | m3        |
        // |           |           | m17         | m8        |
        // |           | w4        |             |           |

        print_matrix_4x4("m8", C11);

        // m5 = (X2 + X11) * (X15 - w3)^T -> C12
        GPU_add(X2, X11, W1_mat);
        GPU_sub(X15, C11, W2_mat);
        GPU_ABt(W1_mat, W2_mat, C12, 1.0, 0.0);
        // | w3        | m5        | z3          |           |
        // |           |           | w5          | m3        |
        // |           |           | m17         | m8        |
        // |           | w4        |             |           |

        print_matrix_4x4("m5", C11);

        // m7 = X11 * w3^T -> C21
        GPU_ABt(X11, C11, C21, 1.0, 0.0);
        // | w3        | m5        | z3          |           |
        // | m7        |           | w5          | m3        |
        // |           |           | m17         | m8        |
        // |           | w4        |             |           |

        print_matrix_4x4("m7", C11);

        // z5 = m5 + m7 + m8 -> C34
        GPU_add(C12, C34, C34);
        GPU_add(C21, C34, C34);
        // | w3        | m5        | z3          |           |
        // | m7        |           | w5          | m3        |
        // |           |           | m17         |           |
        // |           | w4        |             |           |

        print_matrix_4x4("z5", C11);

        // m23 = X1 * (X13 - X5 + X16)^T -> C41
        GPU_sub(X13, X5, W1_mat);
        GPU_add(W1_mat, X16, W1_mat);
        GPU_ABt(X1, W1_mat, C41, 1.0, 0.0);
        // | w3        | m5        | z3          |           |
        // | m7        |           | w5          | m3        |
        // |           |           | m17         | z5        |
        // | m23        | w4       |             |           |

        print_matrix_4x4("m23", C11);

        // w10 = X6 - X7 -> C22  // TODO: better to put somewhere else, requires additional operation
        GPU_sub(X6, X7, C22);
        // | w3        | m5        | z3          |           |
        // | m7        | w10       | w5          | m3        |
        // |           |           | m17         | z5        |
        // | m23        | w4        |             |           |

        print_matrix_4x4("w10", C11);

        // m11 = (X5 + w10) * X5^T -> C32
        GPU_add(X5, C22, W1_mat);
        GPU_ABt(W1_mat, X5, C32, 1.0, 0.0);
        // | w3        | m5        | z3          |           |
        // | m7        | w10       | w5          | m3        |
        // |           | m11       | m17         | z5        |
        // | m23        | w4        |             |           |

        print_matrix_4x4("m11", C11);

        // w2 = X1 - X5 - X6 -> C31
        GPU_sub(X1, X5, W1_mat);
        GPU_sub(W1_mat, X6, C31);
        // | w3        | m5        | z3          |           |
        // | m7        | w10       | w5          | m3        |
        // | w2        | m11       | m17         | z5        |
        // | m23        | w4        |             |           |

        print_matrix_4x4("w2", C11);

        // m2 = (w2 + X7) * (X15 + X5)^T -> C43
        GPU_add(C31, X7, W1_mat);
        GPU_add(X15, X5, W2_mat);
        GPU_ABt(W1_mat, W2_mat, C43, 1.0, 0.0);
        // | w3        | m5        | z3          |           |
        // | m7        | w10       | w5          | m3        |
        // | w2        | m11       | m17         | z5        |
        // | m23        | w4        | m2          |           |

        print_matrix_4x4("m2", C11);

        // z4 = m2 + m11 + m23 -> C41
        GPU_add(C43, C41, C41);
        GPU_add(C32, C41, C41);
        // | w3        | m5        | z3        |           |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        |           |           |

        print_matrix_4x4("z4", C11);

        // m2 -> C12
        GPU_sub(C43, C12, C12);

        print_matrix_4x4("m2", C11);

        // w9 = X7 - X11 -> C43
        GPU_sub(X7, X11, C43);
        // | w3        | m2 - m5   | z3        |           |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        |           |

        print_matrix_4x4("w9", C11);

        // m13 = (-w2 + X3 - w9) * X15^T -> C44
        GPU_sub(X3, C31, W1_mat);
        GPU_sub(W1_mat, C43, W1_mat);
        GPU_ABt(W1_mat, X15, C44, 1.0, 0.0);
        // | w3        | m2 - m5   | z3        |           |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        | m13       |

        print_matrix_4x4("m13", C11);

        // C14 = z4 - z3 - z5 + m13 -> C14
        GPU_sub(C41, C13, C14);
        GPU_sub(C14, C34, C14);
        GPU_add(C14, C44, C14);
        // | w3        | m2 - m5   | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        | m13       |

        print_matrix_4x4("C14", C11);

        // m13 -> C12
        GPU_add(C12, C44, C12);
        // | w3        | m2-m5+m13 | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        |           |

        print_matrix_4x4("C14", C11);

        // w11 = X2 - X3 -> C44
        GPU_sub(X2, X3, C44);
        // | w3        | m2-m5+m13 | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        | w11       |

        print_matrix_4x4("C14", C11);

        // m19 = (-w11) * (-X15 + X7 + X8)^T -> C12
        GPU_sub(X7, X15, W1_mat);
        GPU_add(W1_mat, X8, W1_mat);
        GPU_ABt(C44, W1_mat, W2_mat, -1.0, 0.0);
        GPU_add(W2_mat, C12, C12);
        // C12 = m2 - m5 - z1 + m13 + m19
        // | w3        | C12 + z1  | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        | w11       |

        print_matrix_4x4("m19", C11);

        // m12 = (w11 + X4) * X8^T -> C44
        GPU_add(C44, X4, W1_mat);
        GPU_ABt(W1_mat, X8, C44, 1.0, 0.0);
        // | w3        | C12 + z1  | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | m11       | m17       | z5        |
        // | z4        | w4        | w9        | m12       |

        print_matrix_4x4("m12", C11);

        // z1 = m7 - m11 - m12 -> C32
        GPU_sub(C21, C32, C32);
        GPU_sub(C32, C44, C32);
        // | w3        | C12 + z1  | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | z1        | m17       | z5        |
        // | z4        | w4        | w9        | m12       |

        print_matrix_4x4("z1", C11);
        
        // z1 -> C12
        GPU_sub(C12, C32, C12);
        // | w3        | --------- | z3        | --------- |
        // | m7        | w10       | w5        | m3        |
        // | w2        | z1        | m17       | z5        |
        // | z4        | w4        | w9        | m12       |

        print_matrix_4x4("C12", C11);

        // m22 = (-w10) * (X5 + w9)^T -> C22
        GPU_add(X5, C43, W1_mat);
        GPU_ABt(C22, W1_mat, W2_mat, -1.0, 0.0);
        // TODO: remove excess operation: move m22 to C22
        GPU_add(C22, W2_mat, C22, 0.0, 1.0);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22       | w5        | m3        |
        // | w2        | z1        | m17       | z5        |
        // | z4        | w4        |           | m12       |
        // w9 is not needed anymore

        print_matrix_4x4("m22", C11);
        
        // TODO: try to win an operation here by using y1
        // m18 = X9 * (X13 - X14)^T -> C43
        GPU_sub(X13, X14, W1_mat);
        GPU_ABt(X9, W1_mat, C43, 1.0, 0.0);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22       | w5        | m3        |
        // | w2        | z1        | m17       | z5        |
        // | z4        | w4        | m18       | m12       |

        print_matrix_4x4("m18", C11);

        // z8 = m17 + m18 -> C33
        GPU_add(C33, C43, C33);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22       | w5        | m3        |
        // | w2        | z1        | z8        | z5        |
        // | z4        | w4        | m18       | m12       |

        print_matrix_4x4("z8", C11);

        // Calculate m25 = (X9 + X2 + X10) * X14^T (only needed once)
        GPU_add(X9, X2, W1_mat);
        GPU_add(W1_mat, X10, W1_mat);
        GPU_ABt(W1_mat, X14, W2_mat, 1.0, 0.0);
        GPU_add(W2_mat, C34, C34);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22       | w5        | m3        |
        // | w2        | z1        | z8        | z5 + m25  |
        // | z4        | w4        | m18       | m12       |

        print_matrix_4x4("m25", C11);

        // C34 = z5 + m25 + m3 + z8
        GPU_add(C34, C24, C34);
        GPU_add(C34, C33, C34);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22       | w5        |           |
        // | w2        | z1        | z8        | --------- |
        // | z4        | w4        | m18       | m12       |
        // m3 is not needed anymore

        print_matrix_4x4("C34", C11);

        // move z1 to C22
        GPU_sub(C22, C32, C22);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22 - z1  | w5        |           |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12       |
        // z1 is not needed anymore

        print_matrix_4x4("C22", C11);

        // w1 = X2 + X4 - X8 -> C32
        GPU_add(X2, X4, C32);
        GPU_sub(C32, X8, C32);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22 - z1  | w5        |           |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12       |

        print_matrix_4x4("w1", C11);

        // m1 = (-w1 + X3) * (X8 + X11) -> C24
        GPU_sub(X3, C32, W1_mat);
        GPU_add(X8, X11, W2_mat);
        GPU_ABt(W1_mat, W2_mat, C24, 1.0, 0.0);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m22 - z1  | w5        | m1        |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12       |

        print_matrix_4x4("m1", C11);

        // add m1 to C44 (to calculate z2 later) and C22
        GPU_add(C22, C24, C22);
        GPU_add(C44, C24, C44);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m1+m22-z1 | w5        |           |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // m1 is not needed anymore

        print_matrix_4x4("C22, C44", C11);

        // m6 = (X6 + X11) * (w3 - X11)^T -> C24
        GPU_add(X6, X11, W1_mat);
        GPU_sub(C11, X11, W2_mat);
        GPU_ABt(W1_mat, W2_mat, C24, 1.0, 0.0);
        // | w3        | --------- | z3        | --------- |
        // | m7        | m1+m22-z1 | w5        | m6        |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |

        print_matrix_4x4("m6", C11);

        // add m6 to C21 (to calculate z7) and C22
        GPU_sub(C24, C21, C21);
        GPU_add(C22, C24, C22);
        // C22 = m1 + m6 - z1 + m10 + m22
        // | w3        | --------- | z3        | --------- |
        // | m6 - m7   | C22 - m10 | w5        |           |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // m6 is not needed anymore

        print_matrix_4x4("m6 move", C11);

        // m10 = (w1 - X3 + X7 + X11) * X11^T -> C24
        GPU_sub(C32, X3, W1_mat);
        GPU_add(W1_mat, X7, W1_mat);
        GPU_add(W1_mat, X11, W1_mat);
        GPU_ABt(W1_mat, X11, C24, 1.0, 0.0);
        // | w3        | --------- | z3        | --------- |
        // | m6 - m7   | C22 - m10 | w5        | m10       |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |

        print_matrix_4x4("m10", C11);

        // C22 += m10
        GPU_add(C22, C24, C22);
        // | w3        | --------- | z3        | --------- |
        // | m6 - m7   | --------- | w5        | m10       |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        

        // Now, a tricky moment when we hit memory limit. TODO: check if we can avoid this
        // We need to reuse w1 to calculate m15 and further use it for C13
        // w6 is needed for m15 calculation, but also for m9 later. We'll store w6 in additional memory to reuse it
        // w6 = X10 + X11 -> W_1
        GPU_add(X10, X11, W1_mat);

        // w6 + w5 -> W_2
        GPU_add(W1_mat, C23, W2_mat);
        // | w3        | --------- | z3        | --------- |
        // | m6 - m7   | --------- | w5        | m10       |
        // | w2        | w1        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [ w6 ] [ w6 + w5]

        // m15 = w1 * (w6 + w5)^T -> C23
        GPU_ABt(C32, W2_mat, C23, 1.0, 0.0);
        // | w3        | --------- | z3        | --------- |
        // | m6 - m7   | --------- | m15       | m10       |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [ w6 ]
        // w1 is not needed anymore

        // C13 += m15
        GPU_add(C13, C23, C13);
        // | w3        | --------- | z3 + m15  | --------- |
        // | m6 - m7   | --------- |           | m10       |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [ w6 ]
        // m15 is not needed anymore

        // w7 = X9 + y1 = X9 + X13 - X14 -> W2
        // TODO: reuse y1 here if possible 
        GPU_add(X9, X13, W2_mat);
        GPU_sub(W2_mat, X14, W2_mat);
        // | w3        | --------- | z3 + m15  | --------- |
        // | m6 - m7   | --------- |           | m10       |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [ w6 ] [ w7 ]

        // m9 = X6 * (w7 - w6 + w3) -> C32
        GPU_sub(W2_mat, W1_mat, W1_mat);
        GPU_add(C11, W1_mat, W1_mat);
        GPU_ABt(X6, W1_mat, C32, 1.0, 0.0);
        // |           | --------- | z3 + m15  | --------- |
        // | m6 - m7   | --------- |           | m10       |
        // | w2        | m9        | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [  ] [ w7 ]
        // w3, w6 are not needed anymore

        print_matrix_4x4("m9", C11);

        // z7 = m6 - m7 - m9 -> C23
        GPU_sub(C21, C32, C23);
        // |           | --------- | z3 + m15  | --------- |
        // |           | --------- | z7        | m10       |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [  ] [ w7 ]
        // m9 is not needed anymore

        print_matrix_4x4("z7", C11);

        // w8 = X9 - X8 -> C21
        GPU_sub(X9, X8, C21);
        // |           | --------- | z3 + m15  | --------- |
        // | w8        | --------- | z7        | m10       |
        // | w2        |           | z8        | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [  ] [ w7 ]

        print_matrix_4x4("w8", C11);

        // C33 += z7
        GPU_add(C33, C23, C33);
        // |           | --------- | z3 + m15  | --------- |
        // | w8        | --------- | z7        | m10       |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [  ] [ w7 ]

        // m21 = X8 * (X12 + w8)^T -> C11
        GPU_add(X12, C21, W1_mat);
        GPU_ABt(X8, W1_mat, C11, 1.0, 0.0);
        // | m21       | --------- | z3 + m15  | --------- |
        // | w8        | --------- | z7        | m10       |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       | m12 + m1  |
        // [  ] [ w7 ]

        print_matrix_4x4("m21", C11);

        // z2 = m1 + m12 + m21 -> C44
        GPU_add(C44, C11, C44);
        // |           | --------- | z3 + m15  | --------- |
        // | w8        | --------- | z7        | m10       |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       | z2        |
        // [  ] [ w7 ]
        // m21 is not needed anymore

        print_matrix_4x4("z2", C11);

        // C13 += z2
        GPU_add(C13, C44, C13);
        // |           | --------- | z3+m15+z2 | --------- |
        // | w8        | --------- | z7        | m10       |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       | z2        |
        // [  ] [ w7 ]

        // C23 += z2 + m10
        GPU_add(C23, C44, C23);
        GPU_add(C23, C24, C23);
        // |           | --------- | z3+m15+z2 | --------- |
        // | w8        | --------- | z2+z7+m10 |           |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       |           |
        // [  ] [ w7 ]
        // z2, m10 are not needed anymore

        print_matrix_4x4("C23", C11);

        // m20 = (X5 + w8) * X9^T -> C11
        GPU_add(X5, C21, C21);
        GPU_ABt(C21, X9, C11, 1.0, 0.0);
        // | m20       | --------- | z3+m15+z2 | --------- |
        // |           | --------- | z2+z7+m10 |           |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       |           |
        // [  ] [ w7 ]
        // w8 is not needed anymore

        print_matrix_4x4("m20", C11);

        // m4 = (X9 - X6) * w7^T -> C21
        GPU_sub(X9, X6, W1_mat);
        GPU_ABt(W1_mat, W2_mat, C21, 1.0, 0.0);
        // | m20       | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | z2+z7+m10 |           |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        | m18       |           |
        // [  ] [ w7 ]

        print_matrix_4x4("m4", C11);

        // z6 = m4 - m18 - m20 -> C24
        GPU_sub(C21, C43, C24);
        GPU_sub(C24, C11, C24);
        // |           | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | z2+z7+m10 | z6        |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]
        // m20 is not needed anymore
        // m18 is not needed anymore

        print_matrix_4x4("z6", C11);

        // C23 = z2 - z6 + z7 + m10
        GPU_sub(C23, C24, C23);
        // |           | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | --------- | z6        |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]

        print_matrix_4x4("C23", C11);

        // m26 = (X6 + X10 + X12) * X10^T -> C11
        GPU_add(X6, X10, C44);
        GPU_add(C44, X12, C44);
        GPU_ABt(C44, X10, C11, 1.0, 0.0);
        // | m26       | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | --------- | z6        |
        // | w2        |           | z8 + z7   | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]

        print_matrix_4x4("m26", C11);

        // C33 = m4 - z7 - z8 + m26
        GPU_sub(C21, C33, C33);
        GPU_add(C33, C11, C33);

        // | m26       | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | --------- | z6        |
        // | w2        |           | --------- | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]

        print_matrix_4x4("C33", C11);

        // m16 = (X1 - X8) * (X9 - X16)^T -> C11
        GPU_sub(X1, X8, W1_mat);
        GPU_sub(X9, X16, C32);
        GPU_ABt(W1_mat, C32, C11, 1.0, 0.0);
        // | m16       | --------- | z3+m15+z2 | --------- |
        // | m4        | --------- | --------- | z6        |
        // | w2        |           | --------- | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]

        print_matrix_4x4("m16", C11);

        // C13 += m16
        GPU_add(C13, C11, C13);
        // | m16       | --------- | --------- | --------- |
        // | m4        | --------- | --------- | z6        |
        // | w2        |           | --------- | --------- |
        // | z4        | w4        |           |           |
        // [  ] [ w7 ]
        print_matrix_4x4("C13", C11);

        // m14 = (-w2) * (w7 + w4) -> C21
        GPU_add(C42, W2_mat, C43);
        GPU_ABt(C31, C43, C21, -1.0, 0.0);
        // | m16       | --------- | --------- | --------- |
        // | m14       | --------- | --------- | z6        |
        // | w2        |           | --------- | --------- |
        // | z4        | w4        |           |           |
        // w7 is not needed anymore

        print_matrix_4x4("m14", C11);

        // C24 = z4 + z6 + m14 + m16
        GPU_add(C24, C11, C24);
        GPU_add(C24, C21, C24);
        GPU_add(C24, C41, C24);
        // | m16       | --------- | --------- | --------- |
        // | m14       | --------- | --------- | --------- |
        // | w2        |           | --------- | --------- |
        // | z4        | w4        |           |           |

        print_matrix_4x4("C24", C11);
        
        // Recursively calculate C11
        rtxx(X1.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X2.data, C31, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);

        GPU_add(C21, C31, C11);

        rtxx(X3.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11);

        rtxx(X4.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11);

        // Recursively calculate C44
        rtxx(X13.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X14.data, C31, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C31, C44);

        rtxx(X15.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44);

        rtxx(X16.data, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44);

        cudaFree(Xt);
    }

    // Free all temporary matrices
    cudaFree(W_1); cudaFree(W_2);

    /* dynamic peeling fix-up */
    int pxa = XA % 2;
    int pya = YA % 2;
    int pxc = XC % 2;
    int pyc = YC % 2;

    int nxa = XA - pxa;
    int nya = YA - pya;
    int nxc = XC - pxc;
    int nyc = YC - pyc;

    // Create views for the remaining parts
    Matrix a12 = A_mat.view(nxa, 0, pxa, nya);
    Matrix a21 = A_mat.view(0, nya, nxa, pya);
    Matrix c21 = C_mat.view(0, nyc, nxc, pyc);
    Matrix c11 = C_mat.view(0, 0, nxc, nyc);

    // Final matrix operations
    GPU_ABt(a12, A_mat, c21, 1.0, 0.0);        // (c21 c22) = (a12 a22)t * A
    GPU_ABt(a21, a21, c11, 1.0, 1.0);  // C11 = a21t * a21 + C11
}
