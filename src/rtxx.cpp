#include <cstdio>
#include <cuda_runtime_api.h>

#include "strassen.cpp"

cublasHandle_t handle;


void GPU_T(Float *A, Float *C, int lda, int ldc,
    int XA, int YA) {
  Float one = 1.0;
  Float zero = 0.0;
  cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, XA, YA, &one, A, lda, &zero, C, ldc, C, ldc);
}

void GPU_AtB(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta) {
  cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void GPU_AtB_strassen(Float *A, Float *B, Float *C, Float *A2t,
    int lda, int ldb, int ldc, int ldt,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta, int depth) {
  GPU_T(A, A2t, lda, ldt, YA, XA);
  strassen(A2t, B, C, ldt, ldb, ldc, YA, XB, XC, XA, YB, YC, depth - 1); 
}

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

void print_matrix_4x4(Float *A, int lda) {
    Float h_A[16];  // Host array to store the 4x4 elements
    cudaMemcpy(h_A, A, 16 * sizeof(Float), cudaMemcpyDeviceToHost);
    
    printf("First 4x4 elements of matrix:\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%8.3f ", h_A[i + j * lda]);  // Note: using column-major order
        }
        printf("\n");
    }
    printf("\n");
}

void GPU_ABt_strassen(Float *A, Float *B, Float *C, Float *Xt,
    int lda, int ldb, int ldc, int ldt,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta, int depth) {
  GPU_T(B, Xt, ldb, ldt, YB, XB);
  strassen(A, Xt, C, lda, ldt, ldc, XA, XB, XC, YA, YB, YC, depth); 
}

/*
  lda, ldc is the width in actual memory.
  XA, XC is the width for computation.
  Returns the lower triangular part of C.
  A = XA x YA
  C = XC x YC
*/
void rtxx(Float *A, Float *C, int lda, int ldc,
  int XA, int XC, int YA, int YC, int depth) {
  int XA4 = XA / 4;
  int XC4 = XC / 4;

  int YA4 = YA / 4;
  int YC4 = YC / 4;

  // printf("Debug dimensions:\n");
  // printf("XA=%d, XC=%d, YA=%d, YC=%d\n", XA, XC, YA, YC);
  // printf("XA4=%d, XC4=%d, YA4=%d, YC4=%d\n", XA4, XC4, YA4, YC4);
  // printf("Memory required for W_1: %lu bytes\n", (size_t)XC4 * YC4 * sizeof(Float));

  Float *W_1, *W_2, *W_3, *W_4, *W_5, *W_6;
  Float *m1, *m2, *m3, *m4, *m5, *m6, *m7, *m8, *m9, *m10, *m11, *m12, *m13, *m14, *m15, *m16, *m17, *m18, *m19, *m20, *m21, *m22, *m23, *m24, *m25, *m26;
  int ldw = XC4;
  int ldm = XC4;  // Leading dimension for m matrices

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

  Float *X1, *X2, *X3, *X4, *X5, *X6, *X7, *X8, *X9, *X10, *X11, *X12, *X13, *X14, *X15, *X16;
  
  // Note that notation is inverted from the paper as we are filling the lower triangular part of C here
  Float *C11, *C21, *C31, *C41, *C22, *C32, *C42, *C33, *C43, *C44;

  X1 = A;
  X2 = A + dXA;
  X3 = A + 2 * dXA;
  X4 = A + 3 * dXA;
  X5 = A + dYA;
  X6 = A + dXA + dYA;
  X7 = A + 2 * dXA + dYA;
  X8 = A + 3 * dXA + dYA;
  X9 = A + 2 * dYA;
  X10 = A + dXA + 2 * dYA;
  X11 = A + 2 * dXA + 2 * dYA;
  X12 = A + 3 * dXA + 2 * dYA;
  X13 = A + 3 * dYA;
  X14 = A + dXA + 3 * dYA;
  X15 = A + 2 * dXA + 3 * dYA;
  X16 = A + 3 * dXA + 3 * dYA;
  
  C11 = C;
  C21 = C + dYC;
  C22 = C + dXC + dYC;
  C31 = C + 2 * dYC;
  C32 = C + dXC + 2 * dYC;
  C33 = C + 2 * dXC + 2 * dYC;
  C41 = C + 3 * dYC;
  C42 = C + dXC + 3 * dYC;
  C43 = C + 2 * dXC + 3 * dYC;
  C44 = C + 3 * dXC + 3 * dYC;
  

  /* cutoff criteria */
  float mm = (float)CUTOFF / XA4;
  float nn = (float)CUTOFF / YA4;
  bool stop = (mm + nn) >= 2;

  if (depth <= 1 || stop) {
    // GPU_AtB(A, A, C, lda, lda, ldc, XA, YA, YA, YA, XA, YA, 1.0, 0.0);
    // printf("C matrix expected result:\n");
    // print_matrix_4x4(C, ldc);

    Float *X_i;
    Float *X_j;
    Float *C_curr;

    // Fill in lower left triangle of the matrix
    for(int i=0; i<4; i++) {
      for(int j=0; j<4; j++) {
        C_curr = C + i * dXC + j * dYC;

        for(int k=0; k<4; k++) {
          X_i = A + i * dXA + k * dYA;
          X_j = A + j * dXA + k * dYA;

          if (k == 0) {
            GPU_ABt(X_i, X_j, W_1, lda, lda, ldw, XA4, YA4, YA4, YA4, XA4, YA4, 1.0, 0.0);
          }
          else if(k == 1) {
            GPU_ABt(X_i, X_j, W_2, lda, lda, ldw, XA4, YA4, YA4, YA4, XA4, YA4, 1.0, 0.0);
            GPU_add(W_1, W_2, C_curr, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
          } 
          else {
            GPU_ABt(X_i, X_j, W_1, lda, lda, ldw, XA4, YA4, YA4, YA4, XA4, YA4, 1.0, 0.0);
            GPU_add(C_curr, W_1, C_curr, ldc, ldw, ldc, XC4, YC4, 1.0, 1.0);
          }  
        }
      }
    }

    // printf("C matrix:\n");
    // print_matrix_4x4(C, ldc);  // For matrix A with leading dimension lda
  }
  else {
    // Apply RTXX recursively
    Float *Xt;  // To store transposed matrices
    int ldt = YA4;
    cudaMalloc((void **)&Xt, ldt * XA4 * sizeof(Float));

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
    rtxx(X1, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    rtxx(X2, W_2, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(W_1, W_2, C11, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    rtxx(X3, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(C11, W_1, C11, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    rtxx(X4, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(C11, W_1, C11, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);

    // Reallocate memory for next operations
    err = cudaMalloc((void **)&W_1, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to reallocate W_1: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void **)&W_2, ldw * YC4 * sizeof(Float));
    if (err != cudaSuccess) { printf("Failed to reallocate W_2: %s\n", cudaGetErrorString(err)); return; }

    // C44
    rtxx(X13, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    rtxx(X14, W_2, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(W_1, W_2, C44, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    rtxx(X15, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(C44, W_1, C44, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    rtxx(X16, W_1, lda, ldw, XA4, XC4, YA4, YC4, depth - 1);
    GPU_add(C44, W_1, C44, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);

    // Free memory again before next operations
    cudaFree(W_1); cudaFree(W_2);

    // Calculate w1 to use in different ms
    GPU_add(X2, X4, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_1, X8, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);  // w1 = X2 + X4 - X8
    // occupied: W_1

    // Calculate w6 and w5 to compute m15 using w1 as well
    // But first compute y2, put it to W_3
    GPU_add(X12, X10, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);  // y2 = X12 - X10
    GPU_add(X16, W_3, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);  // w5 = X16 + y2
    GPU_add(X10, X11, W_6, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);  // w6 = X10 + X11
    // occupied: W_1, W_3, W_5, W_6

    // Now m17 = X12 * (-y2)^T
    // GPU_ABt(X12, W_3, m17, lda, lda, ldw, YA4, XA4, XC4, XA4, YA4, YC4, -1.0, 0.0);  // m17
    GPU_ABt_strassen(X12, W_3, m17, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, -1.0, 0.0, depth - 1);
    // calculated: m17
    // occupied: W_1, W_5, W_6
    // W_3 is not needed anymore and can be reused

    // m1 = ((-w1 + X3) * (X8 + X11)^T)^T = (X8 + X11) * (-w1 + X3)^T
    GPU_add(W_1, X3, W_2, ldw, ldw, ldc, XC4, YC4, -1.0, 1.0);  // -w1 + X3
    GPU_add(X8, X11, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);  // X8 + X11
    GPU_ABt_strassen(W_2, W_3, m1, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);  // m1
    
    // calculated: m1, m17
    // occupied: W_1, W_5, W_6

    // m15 = w1 * (w6 + w5)^T
    GPU_add(W_6, W_5, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);  // w6 + w5
    GPU_ABt_strassen(W_1, W_2, m15, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);  // m15
    // calculated: m1, m17, m15
    // occupied: W_1, W_5, W_6

    // m10 = (w1 - X3 + X7 + X11) * X11^T
    GPU_add(W_1, X3, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_2, X7, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_2, X11, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_2, X11, m10, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m10, m17, m15
    // occupied: W_5, W_6
    // W_1 is not needed anymore

    // m3 = (-X2 + X12) * w5^T
    GPU_add(X2, X12, W_1, ldw, ldw, ldc, XC4, YC4, -1.0, 1.0);  // -X2 + X12
    GPU_ABt_strassen(W_1, W_5, m3, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);  // m3
    // calculated: m1, m3, m10, m15, m17
    // occupied: W_5, W_6

    // w3 = X6 + X7
    GPU_add(X6, X7, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: m1, m3, m10, m15, m17
    // occupied: W_3, W_5, W_6

    // m7 = X11 * w3
    GPU_ABt_strassen(X11, W_3, m7, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m7, m10, m15, m17
    // occupied: W_3, W_5, W_6

    // m6 = (X6 + X11) * (w3 - X11)
    GPU_add(X6, X11, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_3, X11, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_ABt_strassen(W_1, W_2, m6, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m6, m7, m10, m15, m17
    // occupied: W_3, W_5, W_6

    // w4 = X14 + X15
    GPU_add(X14, X15, W_4, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: m1, m3, m6, m7, m10, m15, m17
    // occupied: W_3, W_4, W_5, W_6

    // m8 = X2 * (w3 - w4 + w5)^T
    GPU_add(W_3, W_4, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_1, W_5, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(X2, W_1, m8, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m6, m7, m8, m10, m15, m17
    // occupied: W_3, W_4, W_6
    // W_5 is not needed anymore

    // compute y1 = X13 - X14, put it to W_5
    GPU_add(X13, X14, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: m1, m3, m6, m7, m8, m10, m15, m17
    // occupied: W_3, W_4, W_6, W_5 (y1)

    // m18 = X9 * y1^T
    GPU_ABt_strassen(X9, W_5, m18, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m6, m7, m8, m10, m15, m17, m18
    // occupied: W_3, W_4, W_6, W_5 (y1)

    // W_5 = X9 + y1
    GPU_add(X9, W_5, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: m1, m3, m6, m7, m8, m10, m15, m17, m18
    // occupied: W_3, W_4, W_6, W_5
    // W_5 is not needed anymore

    // m4 = (X9 - X6) * w7^T
    GPU_add(X9, X6, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_ABt_strassen(W_1, W_5, m4, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m4, m6, m7, m8, m10, m15, m17, m18
    // occupied: W_3, W_4, W_6, W_5

    // m9 = X6 * (w7 - w6 + w3)^T
    GPU_add(W_5, W_6, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_1, W_3, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(X6, W_1, m9, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m4, m6, m7, m8, m9, m10, m15, m17, m18
    // occupied: W_3, W_4, W_5
    // W_6 is not needed anymore

    // m5 = (X2 + X11) * (X15 - w3)^T
    GPU_add(X2, X11, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(X15, W_3, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_ABt_strassen(W_1, W_2, m5, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m3, m4, m5, m6, m7, m8, m9, m10, m15, m17, m18
    // occupied: W_4, W_5
    // W_3 is not needed anymore

    // w2 = X1 - X5 - X6
    GPU_add(X1, X5, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_2, X6, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: m1, m3, m4, m5, m6, m7, m8, m9, m10, m15, m17, m18
    // occupied: W_2, W_4, W_5

    // m14 = -w2 * (w7 + w4)^T
    GPU_add(W_5, W_4, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_2, W_1, m14, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, -1.0, 0.0, depth - 1);
    // calculated: m1, m3, m4, m5, m6, m7, m8, m9, m10, m14, m15, m17, m18
    // occupied: W_2
    // W_4, W_5 is not needed anymore

    // m2 = (w2 + X7) * (X15 + X5)^T
    GPU_add(W_2, X7, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(X15, X5, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, W_3, m2, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m14, m15, m17, m18
    // occupied: W_2

    // w9 = X7 - X11, put it to W_3
    GPU_add(X7, X11, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m14, m15, m17, m18
    // occupied: W_2, W_3 (w9)

    // m13 = (-w2 + X3 - w9) * X15^T
    GPU_add(W_2, X3, W_1, ldw, ldw, ldc, XC4, YC4, -1.0, 1.0);
    GPU_add(W_1, W_3, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_ABt_strassen(W_1, X15, m13, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m13, m14, m15, m17, m18
    // occupied: W_3 (w9)
    // W_2 is not needed anymore

    // w10 = X6 - X7, put it to W_4
    GPU_add(X6, X7, W_4, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m13, m14, m15, m17, m18
    // occupied: W_3 (w9), W_4 (w10)

    // m22 = -w10 * (X5 + w9)^T
    GPU_add(X5, W_3, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_4, W_1, m22, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, -1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m13, m14, m15, m17, m18, m22
    // occupied: W_4 (w10)
    // W_3 is not needed anymore

    // m11 = (X5 + w10) * X5^T
    GPU_add(X5, W_4, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X5, m11, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m13, m14, m15, m17, m18, m22
    // occupied: none
    // W_4 is not needed anymore

    // w11 = X2 - X3, put it to W_3
    GPU_add(X2, X3, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // occupied: W_3 (w11)

    // m12 = (w11 + X4) * X8^t
    GPU_add(W_3, X4, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X8, m12, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m17, m18, m22
    // occupied: W_3 (w11)

    // m19 = -w11 * (-X15 + X7 + X8)^T
    GPU_add(X15, X7, W_1, ldw, ldw, ldc, XC4, YC4, -1.0, 1.0);
    GPU_add(W_1, X8, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_3, W_1, m19, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, -1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m17, m18, m19, m22
    // occupied: none
    // W_3 is not needed anymore

    // w8 = X9 - X8, put it to W_3
    GPU_add(X9, X8, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m17, m18, m19, m22
    // occupied: W_3 (w8)

    // m20 = (X5 + W8) * X9^T
    GPU_add(X5, W_3, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X9, m20, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m17, m18, m19, m20, m22
    // occupied: W_3 (w8)

    // m21 = X8 * (X12 + X8)^T
    GPU_add(X12, X8, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_3, W_1, m21, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m17, m18, m19, m20, m21, m22
    // occupied: none
    // W_3 is not needed anymore
    
    // m16 = (X1 - X8) * (X9 - X16)^T
    GPU_add(X1, X8, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(X9, X16, W_2, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_ABt_strassen(W_1, W_2, m16, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22
    // occupied: none

    // m23 = X1 * (X13 - X5 + X16)
    GPU_add(X13, X5, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_1, X16, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(X1, W_1, m23, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23
    // occupied: none
    
    // m24 = (-X1 + X4 + X12) * X16^T
    GPU_add(X1, X4, W_1, ldw, ldw, ldc, XC4, YC4, -1.0, 1.0);
    GPU_add(W_1, X12, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X16, m24, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24
    // occupied: none

    // m25 = (X9 + X2 + X10) * X14^T
    GPU_add(X9, X2, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_1, X10, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X14, m25, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25
    // occupied: none

    // m26 = (X6 + X10 + X12) * X10^T
    GPU_add(X6, X10, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_1, X12, W_1, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_ABt_strassen(W_1, X10, m26, Xt, lda, ldw, lda, ldt, YA4, XA4, XC4, XA4, YA4, YC4, 1.0, 0.0, depth - 1);
    // calculated: m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26
    // occupied: none

    // z1 = m7 - m11 - m12, put it to W_3
    GPU_add(m7, m11, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_3, m12, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // occupied: W_3 (z1)

    // c12 = m2 - m5 - z1 + m13 + m19
    GPU_add(m2, m5, C21, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C21, W_3, C21, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C21, m13, C21, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C21, m19, C21, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21
    // occupied: W_3 (z1)

    // C22 = m1 + m6 - z1 + m10 + m22
    GPU_add(m1, m6, C22, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C22, W_3, C22, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C22, m10, C22, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C22, m22, C22, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22
    // occupied: none
    // W_3 is not needed anymore

    // z2 = m1 + m12 + m21, put it to W_3
    GPU_add(m1, m12, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_3, m21, W_3, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22
    // occupied: W_3 (z2)

    // z3 = m3 + m17 - m24, put it to W_4
    GPU_add(m3, m17, W_4, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_4, m24, W_4, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: C21, C22
    // occupied: W_3 (z2), W_4 (z3)

    // C31 = z2 + z3 + m15 + m16
    GPU_add(W_3, W_4, C31, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C31, m15, C31, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C31, m16, C31, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31
    // occupied: W_3 (z2), W_4 (z3)

    // z4 = m2 + m11 + m23, put it to W_5
    GPU_add(m2, m11, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_5, m23, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31
    // occupied: W_3 (z2), W_4 (z3), W_5 (z4)

    // z5 = m5 + m7 + m8, put it to W_6
    GPU_add(m5, m7, W_6, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(W_6, m8, W_6, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31
    // occupied: W_3 (z2), W_4 (z3), W_5 (z4), W_6 (z5)

    // C41 = z4 - z3 - z5 + m13
    GPU_add(W_5, W_4, C41, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C41, W_6, C41, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C41, m13, C41, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31, C41
    // occupied: W_3 (z2), W_5 (z4), W_6 (z5)
    // W_4 is not needed anymore

    // z8 = m17 + m18, put it to W_4
    GPU_add(m17, m18, W_4, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31, C41
    // occupied: W_3 (z2), W_5 (z4), W_6 (z5), W_4 (z8)

    // C43 = m3 + z5 + z8 + m25
    GPU_add(m3, W_6, C43, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C43, W_4, C43, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C43, m25, C43, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31, C41, C43
    // occupied: W_3 (z2), W_5 (z4), W_4 (z8)
    // W_6 is not needed anymore

    // z6 = m4 - m18 - m20, put it to W_6
    GPU_add(m4, m18, W_6, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_6, m20, W_6, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: C21, C22, C31, C41, C43
    // occupied: W_3 (z2), W_5 (z4), W_4 (z8), W_6 (z6)

    // C42 = z4 + z6 + m14 + m16
    GPU_add(W_5, W_6, C42, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C42, m14, C42, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C42, m16, C42, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31, C41, C43, C42
    // occupied: W_3 (z2), W_4 (z8), W_6 (z6)
    // W_5 is not needed anymore

    // z7 = m6 - m7 - m9, put it to W_5
    GPU_add(m6, m7, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(W_5, m9, W_5, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    // calculated: C21, C22, C31, C41, C43, C42
    // occupied: W_3 (z2), W_4 (z8), W_6 (z6), W_5 (z7)

    // C32 = z2 - z6 + z7 + m10
    GPU_add(W_3, W_6, C32, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C32, W_5, C32, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    GPU_add(C32, m10, C32, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    // calculated: C21, C22, C31, C41, C43, C42, C32
    // occupied: W_4 (z8), W_5 (z7)
    // W_6, W_3 are not needed anymore

    // C33 = m4 - z7 - z8 + m26
    GPU_add(m4, W_5, C33, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C33, W_4, C33, ldw, ldw, ldc, XC4, YC4, 1.0, -1.0);
    GPU_add(C33, m26, C33, ldw, ldw, ldc, XC4, YC4, 1.0, 1.0);
    
    cudaFree(Xt);
  }
  cudaFree(W_1);
  cudaFree(W_2);
  cudaFree(W_3);
  cudaFree(W_4);
  cudaFree(W_5);
  cudaFree(W_6);
  cudaFree(m1);
  cudaFree(m2);
  cudaFree(m3);
  cudaFree(m4);
  cudaFree(m5);
  cudaFree(m6);
  cudaFree(m7);
  cudaFree(m8);
  cudaFree(m9);
  cudaFree(m10);
  cudaFree(m11);
  cudaFree(m12);
  cudaFree(m13);
  cudaFree(m14);
  cudaFree(m15);
  cudaFree(m16);
  cudaFree(m17);
  cudaFree(m18);
  cudaFree(m19);
  cudaFree(m20);
  cudaFree(m21);
  cudaFree(m22);
  cudaFree(m23);
  cudaFree(m24);
  cudaFree(m25);
  cudaFree(m26);
  
  
  /* dynamic peeling fix-up */
  int pxa = XA % 2;
  int pya = YA % 2;
  int pxc = XC % 2;
  int pyc = YC % 2;

  int nxa = XA - pxa;
  int nya = YA - pya;
  int nxc = XC - pxc;
  int nyc = YC - pyc;

  Float *a12, *a21;
  Float *c21;
  int dxa = nxa;
  int dya = nya * lda;
  // int dxc = nxc;
  int dyc = nyc * ldc;

  a12 = A + dxa;
  a21 = A + dya;
  // a22 = A + dxa + dya;
  // c12 = C + dxc;
  c21 = C + dyc;
  // c22 = C + dxc + dyc;

  /*
    A11 = nxa x nya
    a12 = pxa x nya
    a21 = nxa x pya
    a22 = pxa x pya
  */
  GPU_AtB(a12, A, c21, lda, lda, ldc, YA, XA, XC, pxa, YA, pyc, 1.0, 0.0);        // (c21 c22) = (a12 a22)t * A
  GPU_AtB(a21, a21, C11, lda, lda, ldc, pya, nxa, nxc, nxa, pya, nyc, 1.0, 1.0);  // C11 = a21t * a21 + C11

}
