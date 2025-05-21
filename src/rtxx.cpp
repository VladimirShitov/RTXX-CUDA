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
void GPU_T(Float *A, Float *C, int lda, int ldc, int XA, int YA) {
    Float one = 1.0;
    Float zero = 0.0;
    cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, XA, YA, &one, A, lda, &zero, C, ldc, C, ldc);
}

// Matrix overload for GPU_T
void GPU_T(const Matrix& A, Matrix& C) {
    GPU_T(A.data, C.data, A.ld, C.ld, A.rows, A.cols);
}

void GPU_AtB_strassen(const Matrix& A, const Matrix& B, Matrix& C, Matrix& A2t, int depth) {
    GPU_T(A, A2t);
    strassen(A2t.data, B.data, C.data, A2t.ld, B.ld, C.ld, 
             A2t.rows, B.rows, C.rows, A2t.cols, B.cols, C.cols, depth - 1);
}

void GPU_ABt_strassen(const Matrix& A, const Matrix& B, Matrix& C, Matrix& Xt, int depth) {
    GPU_T(B, Xt);
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

void rtxx(Float *A, Float *C, int lda, int ldc,
          int XA, int XC, int YA, int YC, int depth) {
    int XA4 = XA / 4;
    int XC4 = XC / 4;
    int YA4 = YA / 4;
    int YC4 = YC / 4;

    Float *W_1, *W_2, *W_3, *W_4, *W_5, *W_6;
    Float *m1, *m2, *m3, *m4, *m5, *m6, *m7, *m8, *m9, *m10, *m11, *m12, *m13, *m14, *m15, *m16, *m17, *m18, *m19, *m20, *m21, *m22, *m23, *m24, *m25, *m26;
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
    err = cudaMalloc((void **)&W_3, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate W_3: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&W_4, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate W_4: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&W_5, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate W_5: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&W_6, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate W_6: %s\n", cudaGetErrorString(err)); return; }

    // Allocate m matrices with proper error checking
    err = cudaMalloc((void **)&m1, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m1: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m2, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m2: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m3, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m3: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m4, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m4: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m5, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m5: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m6, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m6: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m7, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m7: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m8, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m8: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m9, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m9: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m10, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m10: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m11, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m11: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m12, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m12: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m13, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m13: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m14, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m14: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m15, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m15: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m16, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m16: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m17, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m17: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m18, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m18: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m19, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m19: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m20, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m20: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m21, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m21: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m22, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m22: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m23, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m23: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m24, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m24: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m25, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m25: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&m26, ldm * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to allocate m26: %s\n", cudaGetErrorString(err)); return; }

    int dXA = XA4;
    int dYA = YA4 * lda;
    int dXC = XC4;
    int dYC = YC4 * ldc;

    // Create matrix views for input matrix A
    Matrix A_mat(A, XA, YA, lda);
    Matrix X1 = A_mat.view(0, 0, XA4, YA4);
    Matrix X2 = A_mat.view(XA4, 0, XA4, YA4);
    Matrix X3 = A_mat.view(2*XA4, 0, XA4, YA4);
    Matrix X4 = A_mat.view(3*XA4, 0, XA4, YA4);
    Matrix X5 = A_mat.view(0, YA4, XA4, YA4);
    Matrix X6 = A_mat.view(XA4, YA4, XA4, YA4);
    Matrix X7 = A_mat.view(2*XA4, YA4, XA4, YA4);
    Matrix X8 = A_mat.view(3*XA4, YA4, XA4, YA4);
    Matrix X9 = A_mat.view(0, 2*YA4, XA4, YA4);
    Matrix X10 = A_mat.view(XA4, 2*YA4, XA4, YA4);
    Matrix X11 = A_mat.view(2*XA4, 2*YA4, XA4, YA4);
    Matrix X12 = A_mat.view(3*XA4, 2*YA4, XA4, YA4);
    Matrix X13 = A_mat.view(0, 3*YA4, XA4, YA4);
    Matrix X14 = A_mat.view(XA4, 3*YA4, XA4, YA4);
    Matrix X15 = A_mat.view(2*XA4, 3*YA4, XA4, YA4);
    Matrix X16 = A_mat.view(3*XA4, 3*YA4, XA4, YA4);

    // Create matrix views for output matrix C
    Matrix C_mat(C, XC, YC, ldc);
    Matrix C11 = C_mat.view(0, 0, XC4, YC4);
    Matrix C21 = C_mat.view(0, YC4, XC4, YC4);
    Matrix C22 = C_mat.view(XC4, YC4, XC4, YC4);
    Matrix C31 = C_mat.view(0, 2*YC4, XC4, YC4);
    Matrix C32 = C_mat.view(XC4, 2*YC4, XC4, YC4);
    Matrix C33 = C_mat.view(2*XC4, 2*YC4, XC4, YC4);
    Matrix C41 = C_mat.view(0, 3*YC4, XC4, YC4);
    Matrix C42 = C_mat.view(XC4, 3*YC4, XC4, YC4);
    Matrix C43 = C_mat.view(2*XC4, 3*YC4, XC4, YC4);
    Matrix C44 = C_mat.view(3*XC4, 3*YC4, XC4, YC4);

    // Create matrix views for temporary matrices
    Matrix W1_mat(W_1, XC4, YC4, ldw);
    Matrix W2_mat(W_2, XC4, YC4, ldw);
    Matrix W3_mat(W_3, XC4, YC4, ldw);
    Matrix W4_mat(W_4, XC4, YC4, ldw);
    Matrix W5_mat(W_5, XC4, YC4, ldw);
    Matrix W6_mat(W_6, XC4, YC4, ldw);

    Matrix m1_mat(m1, XC4, YC4, ldm);
    Matrix m2_mat(m2, XC4, YC4, ldm);
    Matrix m3_mat(m3, XC4, YC4, ldm);
    Matrix m4_mat(m4, XC4, YC4, ldm);
    Matrix m5_mat(m5, XC4, YC4, ldm);
    Matrix m6_mat(m6, XC4, YC4, ldm);
    Matrix m7_mat(m7, XC4, YC4, ldm);
    Matrix m8_mat(m8, XC4, YC4, ldm);
    Matrix m9_mat(m9, XC4, YC4, ldm);
    Matrix m10_mat(m10, XC4, YC4, ldm);
    Matrix m11_mat(m11, XC4, YC4, ldm);
    Matrix m12_mat(m12, XC4, YC4, ldm);
    Matrix m13_mat(m13, XC4, YC4, ldm);
    Matrix m14_mat(m14, XC4, YC4, ldm);
    Matrix m15_mat(m15, XC4, YC4, ldm);
    Matrix m16_mat(m16, XC4, YC4, ldm);
    Matrix m17_mat(m17, XC4, YC4, ldm);
    Matrix m18_mat(m18, XC4, YC4, ldm);
    Matrix m19_mat(m19, XC4, YC4, ldm);
    Matrix m20_mat(m20, XC4, YC4, ldm);
    Matrix m21_mat(m21, XC4, YC4, ldm);
    Matrix m22_mat(m22, XC4, YC4, ldm);
    Matrix m23_mat(m23, XC4, YC4, ldm);
    Matrix m24_mat(m24, XC4, YC4, ldm);
    Matrix m25_mat(m25, XC4, YC4, ldm);
    Matrix m26_mat(m26, XC4, YC4, ldm);

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
        // Apply RTXX recursively
        Float *Xt;
        int ldt = YA4;
        cudaMalloc((void **)&Xt, ldt * XA4 * sizeof(Float));
        Matrix Xt_mat(Xt, YA4, XA4, ldt);

        // Free all temporary matrices before recursive calls
        cudaFree(W_1); cudaFree(W_2); cudaFree(W_3); cudaFree(W_4);
        cudaFree(W_5); cudaFree(W_6);
        cudaFree(m1); cudaFree(m2); cudaFree(m3); cudaFree(m4);
        cudaFree(m5); cudaFree(m6); cudaFree(m7); cudaFree(m8);
        cudaFree(m9); cudaFree(m10); cudaFree(m11); cudaFree(m12);
        cudaFree(m13); cudaFree(m14); cudaFree(m15); cudaFree(m16);
        cudaFree(m17); cudaFree(m18); cudaFree(m19); cudaFree(m20);
        cudaFree(m21); cudaFree(m22); cudaFree(m23); cudaFree(m24);
        cudaFree(m25); cudaFree(m26);

        // C11
        rtxx(X1.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X2.data, W_2, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(W1_mat, W2_mat, C11);
        rtxx(X3.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C11, W1_mat, C11);
        rtxx(X4.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C11, W1_mat, C11);

        // Reallocate memory for next operations
        err = cudaMalloc((void **)&W_1, ldw * YC4 * sizeof(Float));
        if (err != cudaSuccess) { printf("Failed to reallocate W_1: %s\n", cudaGetErrorString(err)); return; }
        err = cudaMalloc((void **)&W_2, ldw * YC4 * sizeof(Float));
        if (err != cudaSuccess) { printf("Failed to reallocate W_2: %s\n", cudaGetErrorString(err)); return; }

        // C44
        rtxx(X13.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X14.data, W_2, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(W1_mat, W2_mat, C44);
        rtxx(X15.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C44, W1_mat, C44);
        rtxx(X16.data, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C44, W1_mat, C44);

        // Free memory again before next operations
        cudaFree(W_1); cudaFree(W_2);

        // Calculate w1 to use in different ms
        GPU_add(X2, X4, W1_mat);
        GPU_sub(W1_mat, X8, W1_mat);  // w1 = X2 + X4 - X8

        // Calculate w6 and w5 to compute m15 using w1 as well
        // But first compute y2, put it to W_3
        GPU_sub(X12, X10, W3_mat);  // y2 = X12 - X10
        GPU_add(X16, W3_mat, W5_mat);  // w5 = X16 + y2
        GPU_add(X10, X11, W6_mat);  // w6 = X10 + X11

        // Now m17 = X12 * (-y2)^T
        GPU_ABt_strassen(X12, W3_mat, m17_mat, Xt_mat, depth - 1);

        // m1 = ((-w1 + X3) * (X8 + X11)^T)^T = (X8 + X11) * (-w1 + X3)^T
        GPU_sub(W1_mat, X3, W2_mat);  // -w1 + X3
        GPU_add(X8, X11, W3_mat);  // X8 + X11
        GPU_ABt_strassen(W2_mat, W3_mat, m1_mat, Xt_mat, depth - 1);

        // m15 = w1 * (w6 + w5)^T
        GPU_add(W6_mat, W5_mat, W2_mat);  // w6 + w5
        GPU_ABt_strassen(W1_mat, W2_mat, m15_mat, Xt_mat, depth - 1);

        // m10 = (w1 - X3 + X7 + X11) * X11^T
        GPU_sub(W1_mat, X3, W2_mat);
        GPU_add(W2_mat, X7, W2_mat);
        GPU_add(W2_mat, X11, W2_mat);
        GPU_ABt_strassen(W2_mat, X11, m10_mat, Xt_mat, depth - 1);

        // m3 = (-X2 + X12) * w5^T
        GPU_sub(X2, X12, W1_mat);  // -X2 + X12
        GPU_ABt_strassen(W1_mat, W5_mat, m3_mat, Xt_mat, depth - 1);

        // w3 = X6 + X7
        GPU_add(X6, X7, W3_mat);

        // m7 = X11 * w3
        GPU_ABt_strassen(X11, W3_mat, m7_mat, Xt_mat, depth - 1);

        // m6 = (X6 + X11) * (w3 - X11)
        GPU_add(X6, X11, W1_mat);
        GPU_sub(W3_mat, X11, W2_mat);
        GPU_ABt_strassen(W1_mat, W2_mat, m6_mat, Xt_mat, depth - 1);

        // w4 = X14 + X15
        GPU_add(X14, X15, W4_mat);

        // m8 = X2 * (w3 - w4 + w5)^T
        GPU_sub(W3_mat, W4_mat, W1_mat);
        GPU_add(W1_mat, W5_mat, W1_mat);
        GPU_ABt_strassen(X2, W1_mat, m8_mat, Xt_mat, depth - 1);

        // compute y1 = X13 - X14, put it to W_5
        GPU_sub(X13, X14, W5_mat);

        // m18 = X9 * y1^T
        GPU_ABt_strassen(X9, W5_mat, m18_mat, Xt_mat, depth - 1);

        // W_5 = X9 + y1
        GPU_add(X9, W5_mat, W5_mat);

        // m4 = (X9 - X6) * w7^T
        GPU_sub(X9, X6, W1_mat);
        GPU_ABt_strassen(W1_mat, W5_mat, m4_mat, Xt_mat, depth - 1);

        // m9 = X6 * (w7 - w6 + w3)^T
        GPU_sub(W5_mat, W6_mat, W1_mat);
        GPU_add(W1_mat, W3_mat, W1_mat);
        GPU_ABt_strassen(X6, W1_mat, m9_mat, Xt_mat, depth - 1);

        // m5 = (X2 + X11) * (X15 - w3)^T
        GPU_add(X2, X11, W1_mat);
        GPU_sub(X15, W3_mat, W2_mat);
        GPU_ABt_strassen(W1_mat, W2_mat, m5_mat, Xt_mat, depth - 1);

        // w2 = X1 - X5 - X6
        GPU_sub(X1, X5, W2_mat);
        GPU_sub(W2_mat, X6, W2_mat);

        // m14 = -w2 * (w7 + w4)^T
        GPU_add(W5_mat, W4_mat, W1_mat);
        GPU_ABt_strassen(W2_mat, W1_mat, m14_mat, Xt_mat, depth - 1);

        // m2 = (w2 + X7) * (X15 + X5)^T
        GPU_add(W2_mat, X7, W1_mat);
        GPU_add(X15, X5, W3_mat);
        GPU_ABt_strassen(W1_mat, W3_mat, m2_mat, Xt_mat, depth - 1);

        // w9 = X7 - X11, put it to W_3
        GPU_sub(X7, X11, W3_mat);

        // m13 = (-w2 + X3 - w9) * X15^T
        GPU_sub(W2_mat, X3, W1_mat);
        GPU_sub(W1_mat, W3_mat, W1_mat);
        GPU_ABt_strassen(W1_mat, X15, m13_mat, Xt_mat, depth - 1);

        // w10 = X6 - X7, put it to W_4
        GPU_sub(X6, X7, W4_mat);

        // m22 = -w10 * (X5 + w9)^T
        GPU_add(X5, W3_mat, W1_mat);
        GPU_ABt_strassen(W4_mat, W1_mat, m22_mat, Xt_mat, depth - 1);

        // m11 = (X5 + w10) * X5^T
        GPU_add(X5, W4_mat, W1_mat);
        GPU_ABt_strassen(W1_mat, X5, m11_mat, Xt_mat, depth - 1);

        // w11 = X2 - X3, put it to W_3
        GPU_sub(X2, X3, W3_mat);

        // m12 = (w11 + X4) * X8^t
        GPU_add(W3_mat, X4, W1_mat);
        GPU_ABt_strassen(W1_mat, X8, m12_mat, Xt_mat, depth - 1);

        // m19 = -w11 * (-X15 + X7 + X8)^T
        GPU_sub(X15, X7, W1_mat);
        GPU_add(W1_mat, X8, W1_mat);
        GPU_ABt_strassen(W3_mat, W1_mat, m19_mat, Xt_mat, depth - 1);

        // w8 = X9 - X8, put it to W_3
        GPU_sub(X9, X8, W3_mat);

        // m20 = (X5 + W8) * X9^T
        GPU_add(X5, W3_mat, W1_mat);
        GPU_ABt_strassen(W1_mat, X9, m20_mat, Xt_mat, depth - 1);

        // m21 = X8 * (X12 + X8)^T
        GPU_add(X12, X8, W1_mat);
        GPU_ABt_strassen(X8, W1_mat, m21_mat, Xt_mat, depth - 1);

        // m16 = (X1 - X8) * (X9 - X16)^T
        GPU_sub(X1, X8, W1_mat);
        GPU_sub(X9, X16, W2_mat);
        GPU_ABt_strassen(W1_mat, W2_mat, m16_mat, Xt_mat, depth - 1);

        // m23 = X1 * (X13 - X5 + X16)
        GPU_sub(X13, X5, W1_mat);
        GPU_add(W1_mat, X16, W1_mat);
        GPU_ABt_strassen(X1, W1_mat, m23_mat, Xt_mat, depth - 1);

        // m24 = (-X1 + X4 + X12) * X16^T
        GPU_sub(X1, X4, W1_mat);
        GPU_add(W1_mat, X12, W1_mat);
        GPU_ABt_strassen(W1_mat, X16, m24_mat, Xt_mat, depth - 1);

        // m25 = (X9 + X2 + X10) * X14^T
        GPU_add(X9, X2, W1_mat);
        GPU_add(W1_mat, X10, W1_mat);
        GPU_ABt_strassen(W1_mat, X14, m25_mat, Xt_mat, depth - 1);

        // m26 = (X6 + X10 + X12) * X10^T
        GPU_add(X6, X10, W1_mat);
        GPU_add(W1_mat, X12, W1_mat);
        GPU_ABt_strassen(W1_mat, X10, m26_mat, Xt_mat, depth - 1);

        // z1 = m7 - m11 - m12, put it to W_3
        GPU_sub(m7_mat, m11_mat, W3_mat);
        GPU_sub(W3_mat, m12_mat, W3_mat);

        // c12 = m2 - m5 - z1 + m13 + m19
        GPU_sub(m2_mat, m5_mat, C21);
        GPU_sub(C21, W3_mat, C21);
        GPU_add(C21, m13_mat, C21);
        GPU_add(C21, m19_mat, C21);

        // C22 = m1 + m6 - z1 + m10 + m22
        GPU_add(m1_mat, m6_mat, C22);
        GPU_sub(C22, W3_mat, C22);
        GPU_add(C22, m10_mat, C22);
        GPU_add(C22, m22_mat, C22);

        // z2 = m1 + m12 + m21, put it to W_3
        GPU_add(m1_mat, m12_mat, W3_mat);
        GPU_add(W3_mat, m21_mat, W3_mat);

        // z3 = m3 + m17 - m24, put it to W_4
        GPU_add(m3_mat, m17_mat, W4_mat);
        GPU_sub(W4_mat, m24_mat, W4_mat);

        // C31 = z2 + z3 + m15 + m16
        GPU_add(W3_mat, W4_mat, C31);
        GPU_add(C31, m15_mat, C31);
        GPU_add(C31, m16_mat, C31);

        // z4 = m2 + m11 + m23, put it to W_5
        GPU_add(m2_mat, m11_mat, W5_mat);
        GPU_add(W5_mat, m23_mat, W5_mat);

        // z5 = m5 + m7 + m8, put it to W_6
        GPU_add(m5_mat, m7_mat, W6_mat);
        GPU_add(W6_mat, m8_mat, W6_mat);

        // C41 = z4 - z3 - z5 + m13
        GPU_sub(W5_mat, W4_mat, C41);
        GPU_sub(C41, W6_mat, C41);
        GPU_add(C41, m13_mat, C41);

        // z8 = m17 + m18, put it to W_4
        GPU_add(m17_mat, m18_mat, W4_mat);

        // C43 = m3 + z5 + z8 + m25
        GPU_add(m3_mat, W6_mat, C43);
        GPU_add(C43, W4_mat, C43);
        GPU_add(C43, m25_mat, C43);

        // z6 = m4 - m18 - m20, put it to W_6
        GPU_sub(m4_mat, m18_mat, W6_mat);
        GPU_sub(W6_mat, m20_mat, W6_mat);

        // C42 = z4 + z6 + m14 + m16
        GPU_add(W5_mat, W6_mat, C42);
        GPU_add(C42, m14_mat, C42);
        GPU_add(C42, m16_mat, C42);

        // z7 = m6 - m7 - m9, put it to W_5
        GPU_sub(m6_mat, m7_mat, W5_mat);
        GPU_sub(W5_mat, m9_mat, W5_mat);

        // C32 = z2 - z6 + z7 + m10
        GPU_sub(W3_mat, W6_mat, C32);
        GPU_add(C32, W5_mat, C32);
        GPU_add(C32, m10_mat, C32);

        // C33 = m4 - z7 - z8 + m26
        GPU_sub(m4_mat, W5_mat, C33);
        GPU_sub(C33, W4_mat, C33);
        GPU_add(C33, m26_mat, C33);

        cudaFree(Xt);
    }

    // Free all temporary matrices
    cudaFree(W_1); cudaFree(W_2); cudaFree(W_3); cudaFree(W_4);
    cudaFree(W_5); cudaFree(W_6);
    cudaFree(m1); cudaFree(m2); cudaFree(m3); cudaFree(m4);
    cudaFree(m5); cudaFree(m6); cudaFree(m7); cudaFree(m8);
    cudaFree(m9); cudaFree(m10); cudaFree(m11); cudaFree(m12);
    cudaFree(m13); cudaFree(m14); cudaFree(m15); cudaFree(m16);
    cudaFree(m17); cudaFree(m18); cudaFree(m19); cudaFree(m20);
    cudaFree(m21); cudaFree(m22); cudaFree(m23); cudaFree(m24);
    cudaFree(m25); cudaFree(m26);

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
