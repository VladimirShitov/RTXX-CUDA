#include <cublas_v2.h>

#ifdef FLOAT_AS_DOUBLE
typedef double Float;
#define cublasGeam cublasDgeam
#define cublasGemm cublasDgemm
#define cublasSyrk cublasDsyrk
#define CUTOFF 1

#else
typedef float Float;
#define cublasGeam cublasSgeam
#define cublasGemm cublasSgemm
#define cublasSyrk cublasSsyrk
#define CUTOFF 1

#endif  // FLOAT_AS_DOUBLE

void ata(Float *A, Float *C, int lda, int ldc, int XA, int XC, int YA, int YC, int depth);
