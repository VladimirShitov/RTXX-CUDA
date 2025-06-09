#ifndef RTXX_H
#define RTXX_H

#include <cublas_v2.h>

#ifdef FLOAT_AS_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif

// Main rtxx function declaration
void rtxx(Float *A, Float *C, int lda, int ldc,
          int XA, int XC, int YA, int YC, int depth);

// GPU utility functions
void GPU_ABt(Float *A, Float *B, Float *C,
             int lda, int ldb, int ldc,
             int XA, int XB, int XC,
             int YA, int YB, int YC,
             Float alpha, Float beta);

void GPU_T(Float *A, Float *C, int lda, int ldc, int XA, int YA, Float alpha);

void GPU_AtB(Float *A, Float *B, Float *C,
             int lda, int ldb, int ldc,
             int XA, int XB, int XC,
             int YA, int YB, int YC,
             Float alpha, Float beta);

void GPU_add_acc(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int M, int N,
    Float alpha, Float beta,
    Float out_coef = 0.0);

// Fused kernel function declarations
void GPU_A_mul_B_plus_C_t(Float *A, Float *B, Float *C, Float *D,
                          int lda, int ldb, int ldc, int ldd,
                          int M, int N, int K,
                          Float alpha, Float beta, Float gamma,
                          Float out_coef = 0.0);

void GPU_A_plus_B_mul_C_t(Float *A, Float *B, Float *C, Float *D,
                          int lda, int ldb, int ldc, int ldd,
                          int M, int N, int K,
                          Float alpha, Float beta, Float gamma,
                          Float out_coef = 0.0);

void GPU_sum_to_2(Float *A, Float *B, Float *C1, Float *C2,
                  int lda, int ldb, int ldc1, int ldc2,
                  int M, int N,
                  Float alpha, Float beta, Float gamma, Float delta,
                  Float out_coef1 = 0.0, Float out_coef2 = 0.0);

void GPU_sum_3(Float *A, Float *B, Float *C, Float *D,
               int lda, int ldb, int ldc, int ldd,
               int M, int N,
               Float alpha, Float beta, Float gamma,
               Float out_coef = 0.0);

// Global CUBLAS handle
extern cublasHandle_t handle;

#endif // RTXX_H 