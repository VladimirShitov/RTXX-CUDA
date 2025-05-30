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

    Matrix A_mat(A, XA, YA, lda);
    Matrix C_mat(C, XC, YC, ldc);

    /* cutoff criteria */
    float mm = (float)CUTOFF / XA4;
    float nn = (float)CUTOFF / YA4;
    bool stop = (mm + nn) >= 2;

    if (depth <= 1 || stop) {
        GPU_ABt(A_mat, A_mat, C_mat, 1.0, 0.0);

        // Fill in lower left triangle of the matrix
        // for(int i=0; i<4; i++) {
        //     for(int j=0; j<4; j++) {
        //         Matrix C_curr = C_mat.view(i*XC4, j*YC4, XC4, YC4);

        //         for(int k=0; k<4; k++) {
        //             Matrix X_i = A_mat.view(i*XA4, k*YA4, XA4, YA4);
        //             Matrix X_j = A_mat.view(j*XA4, k*YA4, XA4, YA4);

        //             if (k == 0) {
        //                 GPU_ABt(X_i, X_j, W1_mat, 1.0, 0.0);
        //             }
        //             else if(k == 1) {
        //                 GPU_ABt(X_i, X_j, W2_mat, 1.0, 0.0);
        //                 GPU_add(W1_mat, W2_mat, C_curr);
        //             } 
        //             else {
        //                 GPU_ABt(X_i, X_j, W1_mat, 1.0, 0.0);
        //                 GPU_add(C_curr, W1_mat, C_curr);
        //             }  
        //         }
        //     }
        // }
    }
    else {
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

        // We need to store transposed matrix for Strassen calls
        Float *Xt;
        int ldt = YA4;
        cudaMalloc((void **)&Xt, ldt * XA4 * sizeof(Float));
        Matrix Xt_mat(Xt, YA4, XA4, ldt);

        // print_matrix_4x4("Input matrix:", A_mat);

        // Memory layout of the matrices
        // We use column index first to use the same notation as in the paper, but fill lower left triangle
        // | C11       | C21       | C31       | C41       |
        // | C12       | C22       | C32       | C42       |
        // | C13       | C23       | C33       | C43       |
        // | C14       | C24       | C34       | C44       |

        // y2 = X12 - X10 -> C21
        GPU_sub(X12, X10, C21);
        // |                  | y2               |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("y2", C11); // Always use C11 to start with the beginning of the matrix

        // w5 = y2 + X16 -> C41
        GPU_add(C21, X16, C41);
        // |                  | y2               |                  |  w5              |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w5", C11);

        // w9 = X7 - X11 -> C42
        GPU_sub(X7, X11, C42);
        // |                  | y2               |                  | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w9", C11);

        // w10 = X6 - X7 -> C43
        GPU_sub(X6, X7, C43);
        // |                  | y2               |                  | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w10", C11);

        // w2 = X1 - X5 - X6 -> C31
        GPU_sub(X1, X5, C31);
        GPU_sub(C31, X6, C31);
        // |                  | y2               | w2               | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w2", C11);

        // m22 = (-w10) @ (X5 + w9).T -> C22
        GPU_add(X5, C42, C11);
        GPU_ABt(C43, C11, C22, -1.0, 0.0);
        // |                  | y2               | w2               | w5               |
        // |                  | m22              |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("m22", C11);

        // w4 = X14 + X15 -> C32
        GPU_add(X14, X15, C32);
        // |                  | y2               | w2               | w5               |
        // |                  | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w4", C11);

        // m17 = X12 @ (-y2).T -> C34
        GPU_ABt(X12, C21, C34, -1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // |                  | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  | m17              |                  |

        // print_matrix_4x4("m17", C11);

        // m11 = (X5 + w10) @ X5.T -> C12
        GPU_add(X5, C43, C11);
        GPU_ABt(C11, X5, C12);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  | m17              |                  |
        // w10 is not needed anymore and can be rewritten

        // print_matrix_4x4("m11", C11);

        // w3 = X6 + X7 -> C43
        GPU_add(X6, X7, C43);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               | w9               |
        // |                  |                  |                  | w3               |
        // |                  |                  | m17              |                  |

        // print_matrix_4x4("w3", C11);

        // m13 = (-w2 + X3 - w9) @ X15.T -> C14
        GPU_sub(X3, C31, C11);
        GPU_sub(C11, C42, C11);
        GPU_ABt(C11, X15, C14);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // |                  |                  |                  | w3               |
        // | m13              |                  | m17              |                  |
        // w9 is not needed anymore and can be rewritten

        // print_matrix_4x4("m13", C11);

        // m8 = X2 @ (w3 - w4 + w5).T -> C13
        GPU_sub(C43, C32, C11);
        GPU_add(C11, C41, C11);
        GPU_ABt(X2, C11, C13);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               |                  |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m8", C11);

        // m5 = (X2 + X11) @ (X15 - w3).T -> C23
        GPU_add(X2, X11, C11);
        GPU_sub(X15, C43, C44);
        GPU_ABt(C11, C44, C23);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m5", C11);

        // m2 = (w2 + X7) @ (X15 + X5).T -> C21
        GPU_add(C31, X7, C11);
        GPU_add(X15, X5, C44);
        GPU_ABt(C11, C44, C21);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m2", C11);

        // m3 = (-X2 + X12) @ w5.T -> C42
        GPU_sub(X12, X2, C11);
        GPU_ABt(C11, C41, C42);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               | m3               |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m3", C11);

        // w7 = X9 + y1 = X9 + X13 - X14 -> C33
        GPU_sub(X13, X14, C11);
        GPU_add(X9, C11, C33);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               | m3               |
        // | m8               | m5               | w7               | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("w7", C11);

        // m14 = (-w2) @ (w7 + w4).T -> C24
        GPU_add(C33, C32, C11);
        GPU_ABt(C31, C11, C24, -1.0, 0.0);
        // |                  | m2               |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m8               | m5               | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // w2 and w4 are not needed anymore and can be rewritten

        // use m5 to assemble C12 and z5
        GPU_add(C13, C23, C13);
        GPU_sub(C21, C23, C23); // m2 - m5
        GPU_add(C23, C14, C23); // + m13
        // |                  | m2               |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m5 usage", C11);

        // put m11 to m2 to assemble z4 later
        GPU_add(C12, C21, C21);
        // |                  | m2+m11           |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m11 to m2", C11);

        // m7 = X11 @ w3.T -> C31
        GPU_ABt(X11, C43, C31);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m7", C11);

        // w11 = X2 - X3 -> C44 // We'll use slot for intermediate results, but it'll be freed soon
        GPU_sub(X2, X3, C44);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("w11", C11);

        // m12 = (w11 + X4) @ X8.T -> C32
        GPU_add(X4, C44, C11);
        GPU_ABt(C11, X8, C32);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              | m12              | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("m12", C11);

        // z1 = m7 - m11 - m12 -> C12
        GPU_sub(C31, C12, C12); // m7 - m11
        GPU_sub(C12, C32, C12); // - m12
        // |                  | m2+m11           | m7               | w5               |
        // | z1               | m22              | m12              | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("z1", C11);

        // Use z1 to calculate C12 and C22
        GPU_sub(C22, C12, C22); // m22 - z1
        GPU_sub(C23, C12, C23); // m2 - m5 - z1 + m13, only m19 is needed to calculate C12
        // |                  | m2+m11           | m7               | w5               |
        // |                  | m22-z1           | m12              | m3               |
        // | m5+m8            | m2-m5+m13-z1     | w7               | w3               |
        // | m13              | m14              | m17              | w11              |
        // z1 is not needed anymore and can be rewritten

        // print_matrix_4x4("z1 usage", C11);

        // m19 = -w11 @ (-X15 + X7 + X8) -> C12
        GPU_sub(X7, X15, C11);
        GPU_add(C11, X8, C11);
        GPU_ABt(C44, C11, C12, -1.0, 0.0);
        // |                  | m2+m11           | m7               | w5               |
        // | m19              | m22-z1           | m12              | m3               |
        // | m5+m8            | m2-m5+m13-z1     | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // w11 is not needed anymore and can be rewritten
        
        // print_matrix_4x4("m19", C11);

        // C12 = m19 + [C23]
        GPU_add(C23, C12, C12);
        // |                  | m2+m11           | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            |                  | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("C12", C11);

        // m23 = X1 @ (X13 - X5 + X16) -> C23
        GPU_sub(X13, X5, C11);
        GPU_add(C11, X16, C11);
        GPU_ABt(X1, C11, C23);
        // |                  | m2+m11           | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m23              | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m23", C11);

        // z4 = m2 + m11 + m23 -> C21
        GPU_add(C21, C23, C21);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m23              | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // m23 is not needed anymore and can be rewritten

        // print_matrix_4x4("z4", C11);

        // m24 = (-X1 + X4 + X12) @ X16.T -> C23
        GPU_sub(X4, X1, C11);
        GPU_add(C11, X12, C11);
        GPU_ABt(C11, X16, C23);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m24              | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m24", C11);

        // z3 = m3 + m17 - m24 -> C23
        GPU_sub(C34, C23, C23);  // m17 - m24
        GPU_add(C42, C23, C23);  // + m3
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | z3               | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("z3", C11);

        // z5 = m5 + m7 + m8 -> C13
        GPU_add(C31, C13, C13);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | z5               | z3               | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("z5", C11);

        // C14 = z4 - z3 - z5 + m13
        GPU_add(C21, C14, C14);  // z4 + m13
        GPU_sub(C14, C23, C14);  // - z3
        GPU_sub(C14, C13, C14);  // - z5
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | z5               | z3               | w7               | w3               |
        // | C14              | m14              | m17              |                  |

        // print_matrix_4x4("C14", C11);

        // z5 +> m3 to build C34 later
        GPU_add(C42, C13, C42);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // |                  | z3               | w7               | w3               |
        // | ---------------- | m14              | m17              |                  |
        // z5 is not needed anymore and can be rewritten

        // print_matrix_4x4("z5 to m3", C11);

        // Note: losing an operation here
        // m18 = X9 @ y1.T = X9 @ (X13 - X14).T -> C13
        GPU_sub(X13, X14, C11);
        GPU_ABt(X9, C11, C13);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | m17              |                  |

        // print_matrix_4x4("m18", C11);

        // z8 = m17 + m18 -> C34
        GPU_add(C34, C13, C34);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | z8               |                  |

        // print_matrix_4x4("z8", C11);

        // z8 +> C42, almost C34, just m25 is needed
        GPU_add(C42, C34, C42);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | z8               |                  |

        // print_matrix_4x4("z8 to C42", C11);

        // z4 +> C24 to calculate C24 later
        GPU_add(C24, C21, C24);
        // |                  |                  | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               |                  |
        // z4 is not needed anymore and can be rewritten

        // print_matrix_4x4("z4 to C24", C11);

        // w6 = X10 + X11 -> C44
        GPU_add(X10, X11, C44);
        // |                  |                  | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               | w6               |

        // print_matrix_4x4("w6", C11);

        // m9 = X6 @ (w7 - w6 + w3).T -> C21
        GPU_sub(C33, C44, C11);  // w7 - w6
        GPU_add(C11, C43, C11);  // + w3
        GPU_ABt(X6, C11, C21);
        // |                  | m9               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               | w6               |

        // print_matrix_4x4("m9", C11);

        // m9 +> m7 to calculate z7 later; Also +> w6 to w5
        GPU_add(C31, C21, C31);
        GPU_add(C44, C41, C41);
        // |                  | m9               | m7+m9            | w5+w6            |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m9, w6 are not needed anymore and can be rewritten

        // print_matrix_4x4("m9 to m7", C11);

        // m6 = (X6 + X11) @ (w3 - X11).T -> C21
        GPU_add(X6, X11, C11);
        GPU_sub(C43, X11, C44);
        GPU_ABt(C11, C44, C21);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               |                  |
        // | ---------------- | m14+z4           | z8               |                  |
        // w3 is not needed anymore and can be rewritten

        // print_matrix_4x4("m6", C11);

        // m6 +> C22
        GPU_add(C22, C21, C22);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               |                  |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m6 to C22", C11);

        // w1 = X2 + X4 - X8 -> C43
        GPU_add(X2, X4, C43);
        GPU_sub(C43, X8, C43);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("w1", C11);

        // z7 = m6 - (m7+m9)
        GPU_sub(C21, C31, C31);
        // |                  |                  | z7               | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m6 is not needed anymore and can be rewritten

        // print_matrix_4x4("z7", C11);

        // m15 = w1 @ (w6 + w5).T -> C21
        GPU_ABt(C43, C41, C21);
        // |                  | m15              | z7               |                  |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // w5+w6 is not needed anymore and can be rewritten

        // print_matrix_4x4("m15", C11);

        // m15 +> C23 to calculate C13 later
        GPU_add(C23, C21, C23);
        // |                  |                  | z7               |                  |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m15 is not needed anymore and can be rewritten

        // print_matrix_4x4("m15 to C23", C11);

        // m10 = (w1 - X3 + X7 + X11) @ X11.T -> C41
        GPU_sub(C43, X3, C11);
        GPU_add(C11, X7, C11);
        GPU_add(C11, X11, C11);
        GPU_ABt(C11, X11, C41);
        // |                  |                  | z7               | m10              |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m10", C11);

        // m1 = (-w1 + X3) @ (X8 + X11).T -> C21
        GPU_sub(X3, C43, C11);
        GPU_add(X8, X11, C44);
        GPU_ABt(C11, C44, C21); 
        // |                  | m1               | z7               | m10              |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        
        // C22 = m6+m22-z1 + m1 + m10
        GPU_add(C22, C21, C22);
        GPU_add(C22, C41, C22);
        // |                  | m1               | z7               | m10              |
        // | ---------------- | ---------------- | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("C22", C11);
        
        // m12 +> m1 to calculate z2 later
        GPU_add(C21, C32, C21);
        // |                  | m1+m12           | z7               | m10              |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m12 is not needed anymore and can be rewritten

        // w8 = X9 - X8 -> C32
        GPU_sub(X9, X8, C32);
        // |                  | m1+m12           | z7               | m10              |
        // | ---------------- | ---------------- | w8               | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("w8", C11);

        // m21 = X8 @ (X12 + w8).T -> C12 to finish z2 calculation
        GPU_add(X12, C32, C11);
        GPU_ABt(X8, C11, C21, 1.0, 1.0); // Combine with addition
        // |                  | z2               | z7               | m10              |
        // | ---------------- | ---------------- | w8               | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m21", C11);

        // m20 = (X5 + w8) @ X9.T -> m18
        GPU_add(X5, C32, C11);
        GPU_ABt(C11, X9, C13, 1.0, 1.0);
        // |                  | z2               | z7               | m10              |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18+m20          | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // w8 is not needed anymore and can be rewritten

        // print_matrix_4x4("m20", C11);
        
        // Use z2 to calculate C23 and C13 later
        GPU_add(C21, C23, C23);
        GPU_add(C41, C21, C41);
        // |                  |                  | z7               | m10+z2           |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // z2 is not needed anymore and can be rewritten

        // print_matrix_4x4("z2 usage", C11);

        // m16 = (X1 - X8) @ (X9 - X16).T -> C32
        GPU_sub(X1, X8, C11);
        GPU_sub(X9, X16, C44);
        GPU_ABt(C11, C44, C32);
        GPU_add(C32, C24, C24);
        // |                  |                  | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | w7               | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("m16", C11);

        // m4 = (X9 - X6) @ w7.T -> C21
        GPU_sub(X9, X6, C11);
        GPU_ABt(C11, C33, C21);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        |                  | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |
        // w7 is not needed anymore and can be rewritten

        // print_matrix_4x4("m4", C11);

        // m26 = (X6 + X10 + X12) @ X10.T -> C32
        GPU_add(X6, X10, C11);
        GPU_add(C11, X12, C11);
        GPU_ABt(C11, X10, C33);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | m26              | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("m26", C11);

        // z6 = m4 - m18 - m20 -> C13
        GPU_sub(C21, C13, C13);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16                 | m3+z5+z8         |
        // | z6               | z3+m15+z2        | m26              | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("z6", C11);

        // C33 = m4 - z7 - z8 + m26
        GPU_add(C21, C33, C33);  // m26 + m4
        GPU_sub(C33, C31, C33);  // - z7
        GPU_sub(C33, C34, C33);  // - z8
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | z6               | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("C33", C11);

        // C34 = m3 + z5 + z8 + m25
        // m25 = (X9 + X2 + X10) @ X14.T
        GPU_add(X9, X2, C11);
        GPU_add(C11, X10, C11);
        GPU_ABt(C11, X14, C34);
        GPU_add(C34, C42, C34);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              |                  |
        // | z6               | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | m14+z4+m16       | ---------------- |                  |

        // z6 +> C24 and C41
        GPU_add(C24, C13, C24);
        GPU_sub(C41, C13, C41);
        // |                  | m4               | z7               | m10+z2-z6        |
        // | ---------------- | ---------------- | m16              |                  |
        // |                  | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("z6 to C24 and C41", C11);

        // C13 = z2 + z3 + m15 + m16
        GPU_add(C23, C32, C13);
        // |                  | m4               | z7               | m10+z2-z6        |
        // | ---------------- | ---------------- | m16              |                  |
        // | ---------------- |                  | ---------------- | w1               |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("C13", C11);

        // C23 = z2 - z6 + z7 + m10
        GPU_add(C41, C31, C23);
        // |                  |                  |                  |                  |
        // | ---------------- | ---------------- |                  |                  |
        // | ---------------- | ---------------- | ---------------- |                  |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("C23", C11);

        // Recursively calculate C11
        rtxx(X1.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X2.data, C31.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);

        GPU_add(C21, C31, C11);

        rtxx(X3.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11);

        rtxx(X4.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11);

        // Recursively calculate C44
        rtxx(X13.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X14.data, C31.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C31, C44);

        rtxx(X15.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44);

        rtxx(X16.data, C21.data, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44);

        cudaFree(Xt);
    }

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
