#include <cstdio>
#include <cuda_runtime_api.h>

#include "ata.h"
// #include "strassen.cpp"
#include "rtxx.cpp"

/*
  lda, ldc is the width in actual memory.
  XA, XC is the width for computation.
  Returns the lower triangular part of C.
  A = XA x YA
  C = XC x YC
*/
void ata(Float *A, Float *C, int lda, int ldc,
    int XA, int XC, int YA, int YC, int depth) {
  int XA2 = XA / 2;
  int XC2 = XC / 2;

  int YA2 = YA / 2;
  int YC2 = YC / 2;

  Float *W_1, *W_2;
  int ldw = XC2;
  cudaMalloc((void **)&W_1, ldw * YC2 * sizeof(Float));
  cudaMalloc((void **)&W_2, ldw * YC2 * sizeof(Float));

  int dXA = XA2;
  int dYA = YA2 * lda;

  int dXC = XC2;
  int dYC = YC2 * ldc;

  Float *A11, *A12, *A21, *A22;
  Float *C11, *C21, *C22;

  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;

  C11 = C;
  // C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  float mm = (float)CUTOFF / XA2;
  float nn = (float)CUTOFF / YA2;
  bool stop = (mm + nn) >= 2;

  if (depth <= 1 || stop) {
    GPU_AtB(A11, A11, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S1 = A11t * A11
    GPU_AtB(A21, A21, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S2 = A21t * A21
    GPU_add(W_1, W_2, C11, ldw, ldw, ldc, XC2, YC2, 1.0, 1.0);                      // C11 = S1 + S2
    GPU_AtB(A12, A12, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S3 = A12t * A12
    GPU_AtB(A22, A22, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S4 = A22t * A22
    GPU_add(W_1, W_2, C22, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                     // C22 = S3 + S4
    GPU_AtB(A12, A11, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S5 = A12t * A11
    GPU_AtB(A22, A21, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S6 = A22t * A21
    GPU_add(W_1, W_2, C21, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                     // C21 = S5 + S6
  }
  else {
    Float *A2t;
    int ldt = YA2;
    cudaMalloc((void **)&A2t, ldt * XA2 * sizeof(Float));

    ata(A11, W_1, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S1 = ata(A11)
    ata(A21, W_2, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S2 = ata(A21)
    GPU_add(W_1, W_2, C11, ldw, ldw, ldc, XC2, YC2, 1.0, 1.0);                        // C11 = S1 + S2
    ata(A12, W_1, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S3 = ata(A12)
    ata(A22, W_2, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S4 = ata(A22)
    GPU_add(W_1, W_2, C22, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                       // C22 = S3 + S4
    GPU_T(A12, A2t, lda, ldt, YA2, XA2);                                              // A12t
    strassen(A2t, A11, W_1, ldt, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, depth - 1);  // S5 = strassen(A12t, A11)
    GPU_T(A22, A2t, lda, ldt, YA2, XA2);                                              // A22t
    strassen(A2t, A21, W_2, ldt, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, depth - 1);  // S6 = strassen(A22t, A21)
    GPU_add(W_1, W_2, C21, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                       // C21 = S5 + S6

    cudaFree(A2t);
  }
  cudaFree(W_1);
  cudaFree(W_2);

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


int main(int argc, char **argv) {
  if(argc != 7) {
    printf("Usage: %s <M> <N> <iter> <check> <depth> <no_header>\n", argv[0]);
    return -1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int iter = atoi(argv[3]);
  int check = atoi(argv[4]);
  int depth = atoi(argv[5]);
  int no_header = atoi(argv[6]);

  int sizeA = M * N;
  int sizeC = N * N;
  int memSizeA = sizeA * sizeof(Float);
  int memSizeC = sizeC * sizeof(Float);

  Float *h_A = (Float *)malloc(memSizeA);
  Float *ata_C = (Float *)malloc(memSizeC);
  Float *classic_C = (Float *)malloc(memSizeC);
  Float *rtxx_C = (Float *)malloc(memSizeC);

  for (int i = 0; i < sizeA; i++) {
    h_A[i] = (Float)rand() / RAND_MAX;
  }
  for (int i = 0; i < sizeC; i++) {
    ata_C[i] = 0.0;
    classic_C[i] = 0.0;
    rtxx_C[i] = 0.0;
  }

  Float *d_A, *d_C_ata, *d_C_rtxx, *d_C_classic;
  cudaMalloc((void **)&d_A, memSizeA);
  cudaMalloc((void **)&d_C_ata, memSizeC);
  cudaMalloc((void **)&d_C_rtxx, memSizeC);
  cudaMalloc((void **)&d_C_classic, memSizeC);
  cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice);

  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuBLAS initialization error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  CudaTimer ct;

  // ATA algorithm
  cudaMemset(d_C_ata, 0, memSizeC);  // Clear before each algorithm
  ct.start();
  for (int i = 0; i < iter; i++) {
    ata(d_A, d_C_ata, N, N, N, N, M, N, depth);
  }
  ct.stop();

  float ataTime = ct.value() / iter;
  cudaMemcpy(ata_C, d_C_ata, memSizeC, cudaMemcpyDeviceToHost);

  // RTXX algorithm
  cudaMemset(d_C_rtxx, 0, memSizeC);  // Clear before each algorithm
  ct.start();
  for (int i = 0; i < iter; i++) {
    rtxx(d_A, d_C_rtxx, N, N, N, N, M, N, depth);
  }
  ct.stop();

  float rtxxTime = ct.value() / iter;
  cudaMemcpy(rtxx_C, d_C_rtxx, memSizeC, cudaMemcpyDeviceToHost);

  // Classic algorithm
  cudaMemset(d_C_classic, 0, memSizeC);  // Clear before each algorithm
  ct.start();
  for (int i = 0; i < iter; i++) {
    // GPU_AtB(d_A, d_A, d_C_classic, N, N, N, M, N, N, N, M, N, 1.0, 0.0);
    Float alpha = 1.0;
    Float beta = 0.0;
    cublasSyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, N, M, &alpha, d_A, N, &beta, d_C_classic, N);
  }
  ct.stop();

  float classicTime = ct.value() / iter;
  cudaMemcpy(classic_C, d_C_classic, memSizeC, cudaMemcpyDeviceToHost);

  float ata_speedup = classicTime / ataTime;
  float rtxx_speedup = classicTime / rtxxTime;
  
  if (!no_header) {
    printf("M\tN\tdepth\tcuBLAS_time\tAtA_time\tRTXX_time\tATA_speedup\tRTXX_speedup\n");
  }
  printf("%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", 
         M, N, depth, classicTime, ataTime, rtxxTime, ata_speedup, rtxx_speedup);

  Float ata_absErr = 0.0;
  Float rtxx_absErr = 0.0;

  if (check) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        ata_absErr += abs(ata_C[i * N + j] - classic_C[i * N + j]);
        rtxx_absErr += abs(rtxx_C[i * N + j] - classic_C[i * N + j]);
      }
    }
    int numel = N * (N + 1) / 2;
    printf("ATA: Mean absolute error: %g\n", ata_absErr / numel);
    printf("RTXX: Mean absolute error: %g\n", rtxx_absErr / numel);

    // Split matrices into blocks and compute error for each block    
    printf("\nBlock-wise absolute errors:\n");
    printf("Format: (ATA error / RTXX error)\n\n");

    const int N_BLOCKS = 4;

    int block_rows = N / N_BLOCKS;
    int block_cols = N / N_BLOCKS;
    
    for (int bi = 0; bi < N_BLOCKS; bi++) {
      for (int bj = 0; bj < N_BLOCKS; bj++) {
        Float ata_block_err = 0.0;
        Float rtxx_block_err = 0.0;
        int block_elements = 0;
        
        // Compute error for current block
        for (int i = bi * block_rows; i < (bi + 1) * block_rows; i++) {
          for (int j = bj * block_cols; j < (bj + 1) * block_cols; j++) {
            if (j <= i) { // Only lower triangular part
              ata_block_err += abs(ata_C[i * N + j] - classic_C[i * N + j]);
              rtxx_block_err += abs(rtxx_C[i * N + j] - classic_C[i * N + j]);
              block_elements++;
            }
          }
        }
        
        // Normalize by number of elements in block
        if (block_elements > 0) {
          ata_block_err /= block_elements;
          rtxx_block_err /= block_elements;
          printf("(%.3f / %.3f)\t", ata_block_err, rtxx_block_err);
        }
      }
      printf("\n");
    }
    printf("\n");
  }

  free(h_A);
  free(ata_C);
  free(rtxx_C);
  free(classic_C);
  cudaFree(d_A);
  cudaFree(d_C_ata);
  cudaFree(d_C_rtxx);
  cudaFree(d_C_classic);

  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuBLAS shutdown error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }
}
