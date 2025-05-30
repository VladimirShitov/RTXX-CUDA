#include <cstdio>
#include <cuda_runtime_api.h>

#include "strassen.cpp"

cublasHandle_t handle;

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

void GPU_T(Float *A, Float *C, int lda, int ldc, int XA, int YA, Float alpha = 1.0) {
    Float zero = 0.0;
    cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &zero, C, ldc, C, ldc);
}

void GPU_AtB(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta) {
    cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void rtxx(Float *A, Float *C, int lda, int ldc,
          int XA, int XC, int YA, int YC, int depth) {
    int XA4 = XA / 4;
    int XC4 = XC / 4;
    int YA4 = YA / 4;
    int YC4 = YC / 4;

    /* cutoff criteria */
    float mm = (float)CUTOFF / XA4;
    float nn = (float)CUTOFF / YA4;
    bool stop = (mm + nn) >= 2;

    if (depth < 1 || stop) {
        Float alpha = 1.0;
        Float beta = 0.0;
        cublasSyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &beta, C, ldc);
    }
    else {
        Float *X1 = A;
        Float *X2 = A + YA4 * lda;
        Float *X3 = A + 2 * YA4 * lda;
        Float *X4 = A + 3 * YA4 * lda;
        Float *X5 = A + XA4;
        Float *X6 = A + XA4 + YA4 * lda;
        Float *X7 = A + XA4 + 2 * YA4 * lda;
        Float *X8 = A + XA4 + 3 * YA4 * lda;
        Float *X9 = A + 2 * XA4;
        Float *X10 = A + 2 * XA4 + YA4 * lda;
        Float *X11 = A + 2 * XA4 + 2 * YA4 * lda;
        Float *X12 = A + 2 * XA4 + 3 * YA4 * lda;
        Float *X13 = A + 3 * XA4;
        Float *X14 = A + 3 * XA4 + YA4 * lda;
        Float *X15 = A + 3 * XA4 + 2 * YA4 * lda;
        Float *X16 = A + 3 * XA4 + 3 * YA4 * lda;

        Float *C11 = C;
        Float *C12 = C + YC4 * ldc;
        Float *C13 = C + 2 * YC4 * ldc;
        Float *C14 = C + 3 * YC4 * ldc;
        Float *C21 = C + XC4;
        Float *C22 = C + XC4 +  YC4 * ldc;
        Float *C23 = C + XC4 + 2 * YC4 * ldc;
        Float *C24 = C + XC4 + 3 * YC4 * ldc;
        Float *C31 = C + 2 * XC4;
        Float *C32 = C + 2 * XC4 + YC4 * ldc;
        Float *C33 = C + 2 * XC4 + 2 * YC4 * ldc;
        Float *C34 = C + 2 * XC4 + 3 * YC4 * ldc;
        Float *C41 = C + 3 * XC4;
        Float *C42 = C + 3 * XC4 + YC4 * ldc;
        Float *C43 = C + 3 * XC4 + 2 * YC4 * ldc;
        Float *C44 = C + 3 * XC4 + 3 * YC4 * ldc;

        // Memory layout of the matrices
        // We use column index first to use the same notation as in the paper, but fill lower left triangle
        // | C11       | C21       | C31       | C41       |
        // | C12       | C22       | C32       | C42       |
        // | C13       | C23       | C33       | C43       |
        // | C14       | C24       | C34       | C44       |

        // y2 = X12 - X10 -> C21

        GPU_add(X12, X10, C21, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | y2               |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("y2", C11); // Always use C11 to start with the beginning of the matrix

        // w5 = y2 + X16 -> C41
        GPU_add(C21, X16, C41, ldc, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | y2               |                  |  w5              |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w5", C11);

        // w9 = X7 - X11 -> C42
        GPU_add(X7, X11, C42, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | y2               |                  | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  |                  |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w9", C11);

        // w10 = X6 - X7 -> C43
        GPU_add(X6, X7, C43, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | y2               |                  | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w10", C11);

        // w2 = X1 - X5 - X6 -> C31
        GPU_add(X1, X5, C31, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C31, X6, C31, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | y2               | w2               | w5               |
        // |                  |                  |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w2", C11);

        // m22 = (-w10) @ (X5 + w9).T -> C22
        GPU_add(X5, C42, C11, lda, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C43, C11, C22, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, -1.0, 0.0);
        // |                  | y2               | w2               | w5               |
        // |                  | m22              |                  | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("m22", C11);

        // w4 = X14 + X15 -> C32
        GPU_add(X14, X15, C32, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | y2               | w2               | w5               |
        // |                  | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  |                  |                  |

        // print_matrix_4x4("w4", C11);

        // m17 = X12 @ (-y2).T -> C34
        GPU_ABt(X12, C21, C34, lda, ldc, ldc, XA4, XC4, XC4, YA4, YC4, YC4, -1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // |                  | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  | m17              |                  |

        // print_matrix_4x4("m17", C11);

        // m11 = (X5 + w10) @ X5.T -> C12
        GPU_add(X5, C43, C11, lda, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X5, C12, ldc, lda, ldc, XC4, XA4, XC4, YC4, YA4, YC4, 1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               | w9               |
        // |                  |                  |                  | w10              |
        // |                  |                  | m17              |                  |
        // w10 is not needed anymore and can be rewritten

        // print_matrix_4x4("m11", C11);

        // w3 = X6 + X7 -> C43
        GPU_add(X6, X7, C43, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               | w9               |
        // |                  |                  |                  | w3               |
        // |                  |                  | m17              |                  |

        // print_matrix_4x4("w3", C11);

        // m13 = (-w2 + X3 - w9) @ X15.T -> C14
        GPU_add(X3, C31, C11, lda, ldc, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, C42, C11, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, X15, C14, ldc, lda, ldc, XC4, XA4, XC4, YC4, YA4, YC4, 1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // |                  |                  |                  | w3               |
        // | m13              |                  | m17              |                  |
        // w9 is not needed anymore and can be rewritten

        // print_matrix_4x4("m13", C11);

        // m8 = X2 @ (w3 - w4 + w5).T -> C13
        GPU_add(C43, C32, C11, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, C41, C11, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(X2, C11, C13, lda, ldc, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               |                  |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m8", C11);

        // m5 = (X2 + X11) @ (X15 - w3).T -> C23
        GPU_add(X2, X11, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(X15, C43, C44, lda, ldc, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, C44, C23, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, 1.0, 0.0);
        // |                  |                  | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m5", C11);

        // m2 = (w2 + X7) @ (X15 + X5).T -> C21
        GPU_add(C31, X7, C11, ldc, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(X15, X5, C44, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, C44, C21, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, 1.0, 0.0);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               |                  |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m2", C11);

        // m3 = (-X2 + X12) @ w5.T -> C42
        GPU_add(X12, X2, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, C41, C42, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, 1.0, 0.0);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               | m3               |
        // | m8               | m5               |                  | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("m3", C11);

        // w7 = X9 + y1 = X9 + X13 - X14 -> C33
        GPU_add(X13, X14, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(X9, C11, C33, lda, ldc, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m2               | w2               | w5               |
        // | m11              | m22              | w4               | m3               |
        // | m8               | m5               | w7               | w3               |
        // | m13              |                  | m17              |                  |

        // print_matrix_4x4("w7", C11);

        // m14 = (-w2) @ (w7 + w4).T -> C24
        GPU_add(C33, C32, C11, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C31, C11, C24, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, -1.0, 0.0);
        // |                  | m2               |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m8               | m5               | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // w2 and w4 are not needed anymore and can be rewritten

        // use m5 to assemble C12 and z5
        GPU_add(C13, C23, C13, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C21, C23, C23, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0); // m2 - m5
        GPU_add(C23, C14, C23, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0); // + m13
        // |                  | m2               |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m5 usage", C11);

        // put m11 to m2 to assemble z4 later
        GPU_add(C12, C21, C21, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m2+m11           |                  | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m11 to m2", C11);

        // m7 = X11 @ w3.T -> C31
        GPU_ABt(X11, C43, C31, lda, ldc, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m7", C11);

        // w11 = X2 - X3 -> C44 // We'll use slot for intermediate results, but it'll be freed soon
        GPU_add(X2, X3, C44, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              |                  | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("w11", C11);

        // m12 = (w11 + X4) @ X8.T -> C32
        GPU_add(X4, C44, C11, lda, ldc, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X8, C32, ldc, lda, ldc, XC4, XA4, XC4, YC4, YA4, YC4, 1.0, 0.0);
        // |                  | m2+m11           | m7               | w5               |
        // | m11              | m22              | m12              | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("m12", C11);

        // z1 = m7 - m11 - m12 -> C12
        GPU_add(C31, C12, C12, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0); // m7 - m11
        GPU_add(C12, C32, C12, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0); // - m12
        // |                  | m2+m11           | m7               | w5               |
        // | z1               | m22              | m12              | m3               |
        // | m5+m8            | m2-m5+m13        | w7               | w3               |
        // | m13              | m14              | m17              | w11              |

        // print_matrix_4x4("z1", C11);

        // Use z1 to calculate C12 and C22
        GPU_add(C22, C12, C22, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0); // m22 - z1
        GPU_add(C23, C12, C12, ldc, ldc, ldc, XA4, YA4, 1.0, -1.0); // m2 - m5 - z1 + m13, only m19 is needed to calculate C12
        // |                  | m2+m11           | m7               | w5               |
        // | m22-z1           |                  | m12              | m3               |
        // | m5+m8            | m2-m5+m13-z1     | w7               | w3               |
        // | m13              | m14              | m17              | w11              |
        // z1 is not needed anymore and can be rewritten

        // print_matrix_4x4("z1 usage", C11);

        // m19 = -w11 @ (-X15 + X7 + X8) -> C12
        GPU_add(X7, X15, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, X8, C11, ldc, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C44, C11, C12, ldc, ldc, ldc, XC4, XC4, XC4, YC4, YC4, YC4, -1.0, 1.0);
        // |                  | m2+m11           | m7               | w5               |
        // | m19              | m22-z1           | m12              | m3               |
        // | m5+m8            | m2-m5+m13-z1     | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // w11 is not needed anymore and can be rewritten

        // print_matrix_4x4("C12", C11);

        // m23 = X1 @ (X13 - X5 + X16) -> C23
        GPU_add(X13, X5, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, X16, C11, ldc, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(X1, C11, C23, lda, ldc, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m2+m11           | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m23              | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m23", C11);

        // z4 = m2 + m11 + m23 -> C21
        GPU_add(C21, C23, C21, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m23              | w7               | w3               |
        // | m13              | m14              | m17              |                  |
        // m23 is not needed anymore and can be rewritten

        // print_matrix_4x4("z4", C11);

        // m24 = (-X1 + X4 + X12) @ X16.T -> C23
        GPU_add(X4, X1, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, X12, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X16, C23, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | m24              | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("m24", C11);

        // z3 = m3 + m17 - m24 -> C23
        GPU_add(C34, C23, C23, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // m17 - m24
        GPU_add(C42, C23, C23, lda, lda, ldc, XA4, YA4, 1.0, 1.0);  // + m3
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | m5+m8            | z3               | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("z3", C11);

        // z5 = m5 + m7 + m8 -> C13
        GPU_add(C31, C13, C13, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | z5               | z3               | w7               | w3               |
        // | m13              | m14              | m17              |                  |

        // print_matrix_4x4("z5", C11);

        // C14 = z4 - z3 - z5 + m13
        GPU_add(C21, C14, C14, lda, lda, ldc, XA4, YA4, 1.0, 1.0);  // z4 + m13
        GPU_add(C14, C23, C14, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // - z3
        GPU_add(C14, C13, C14, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // - z5
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3               |
        // | z5               | z3               | w7               | w3               |
        // | C14              | m14              | m17              |                  |

        // print_matrix_4x4("C14", C11);

        // z5 +> m3 to build C34 later
        GPU_add(C42, C13, C42, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // |                  | z3               | w7               | w3               |
        // | ---------------- | m14              | m17              |                  |
        // z5 is not needed anymore and can be rewritten

        // print_matrix_4x4("z5 to m3", C11);

        // Note: losing an operation here
        // m18 = X9 @ y1.T = X9 @ (X13 - X14).T -> C13
        GPU_add(X13, X14, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(X9, C11, C13, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | m17              |                  |

        // print_matrix_4x4("m18", C11);

        // z8 = m17 + m18 -> C34
        GPU_add(C34, C13, C34, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5            |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | z8               |                  |

        // print_matrix_4x4("z8", C11);

        // z8 +> C42, almost C34, just m25 is needed
        GPU_add(C42, C34, C42, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | z4               | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14              | z8               |                  |

        // print_matrix_4x4("z8 to C42", C11);

        // z4 +> C24 to calculate C24 later
        GPU_add(C24, C21, C24, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               |                  |
        // z4 is not needed anymore and can be rewritten

        // print_matrix_4x4("z4 to C24", C11);

        // w6 = X10 + X11 -> C44
        GPU_add(X10, X11, C44, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | m7               | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               | w6               |

        // print_matrix_4x4("w6", C11);

        // m9 = X6 @ (w7 - w6 + w3).T -> C21
        GPU_add(C33, C44, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // w7 - w6
        GPU_add(C11, C43, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);  // + w3
        GPU_ABt(X6, C11, C31, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 1.0); // += m7 to calculate z7 later
        // |                  |                  | m7+m9            | w5               |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               | w6               |

        // print_matrix_4x4("m9", C11);

        // w6 +> w5
        GPU_add(C44, C41, C41, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | m7+m9            | w5+w6            |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m9, w6 are not needed anymore and can be rewritten

        // print_matrix_4x4("m9 to m7", C11);

        // m6 = (X6 + X11) @ (w3 - X11).T -> C21
        GPU_add(X6, X11, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C43, X11, C44, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, C44, C21, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m22-z1           | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w3               |
        // | ---------------- | m14+z4           | z8               |                  |
        // w3 is not needed anymore and can be rewritten

        // print_matrix_4x4("m6", C11);

        // m6 +> C22
        GPU_add(C22, C21, C22, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               |                  |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m6 to C22", C11);

        // w1 = X2 + X4 - X8 -> C43
        GPU_add(X2, X4, C43, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C43, X8, C43, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | m6               | m7+m9            | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("w1", C11);

        // z7 = m6 - (m7+m9)
        GPU_add(C21, C31, C31, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  |                  | z7               | w5+w6            |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3               | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m6 is not needed anymore and can be rewritten

        // print_matrix_4x4("z7", C11);

        // m15 = w1 @ (w6 + w5).T -> C21
        GPU_ABt(C43, C41, C23, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 1.0);
        // |                  |                  | z7               |                  |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // w5+w6 is not needed anymore and can be rewritten

        // print_matrix_4x4("m15", C11);

        // print_matrix_4x4("m15 to C23", C11);

        // m10 = (w1 - X3 + X7 + X11) @ X11.T -> C41
        GPU_add(C43, X3, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(C11, X7, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C11, X11, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X11, C41, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  |                  | z7               | m10              |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m10", C11);

        // m1 = (-w1 + X3) @ (X8 + X11).T -> C21
        GPU_add(X3, C43, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(X8, X11, C44, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, C44, C21, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m1               | z7               | m10              |
        // | ---------------- | m6+m22-z1        | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        
        // C22 = m6+m22-z1 + m1 + m10
        GPU_add(C22, C21, C22, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C22, C41, C22, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m1               | z7               | m10              |
        // | ---------------- | ---------------- | m12              | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("C22", C11);
        
        // m12 +> m1 to calculate z2 later
        GPU_add(C21, C32, C21, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m1+m12           | z7               | m10              |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // m12 is not needed anymore and can be rewritten

        // w8 = X9 - X8 -> C32
        GPU_add(X9, X8, C32, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | m1+m12           | z7               | m10              |
        // | ---------------- | ---------------- | w8               | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("w8", C11);

        // m21 = X8 @ (X12 + w8).T -> C12 to finish z2 calculation
        GPU_add(X12, C32, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(X8, C11, C21, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 1.0);
        // |                  | z2               | z7               | m10              |
        // | ---------------- | ---------------- | w8               | m3+z5+z8         |
        // | m18              | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |

        // print_matrix_4x4("m21", C11);

        // m20 = (X5 + w8) @ X9.T -> m18
        GPU_add(X5, C32, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X9, C13, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 1.0);
        // |                  | z2               | z7               | m10              |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18+m20          | z3+m15           | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // w8 is not needed anymore and can be rewritten

        // print_matrix_4x4("m20", C11);
        
        // Use z2 to calculate C23 and C13 later
        GPU_add(C21, C23, C23, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C41, C21, C41, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | z7               | m10+z2           |
        // | ---------------- | ---------------- |                  | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | w7               | w1               |
        // | ---------------- | m14+z4           | z8               |                  |
        // z2 is not needed anymore and can be rewritten

        // print_matrix_4x4("z2 usage", C11);

        // m16 = (X1 - X8) @ (X9 - X16).T -> C32
        GPU_add(X1, X8, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_add(X9, X16, C44, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, C44, C32, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        GPU_add(C32, C24, C24, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | w7               | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("m16", C11);

        // m4 = (X9 - X6) @ w7.T -> C21
        GPU_add(X9, X6, C11, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        GPU_ABt(C11, C33, C21, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        |                  | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |
        // w7 is not needed anymore and can be rewritten

        // print_matrix_4x4("m4", C11);

        // m26 = (X6 + X10 + X12) @ X10.T -> C32
        GPU_add(X6, X10, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C11, X12, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X10, C33, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | m18+m20          | z3+m15+z2        | m26              | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("m26", C11);

        // z6 = m4 - m18 - m20 -> C13
        GPU_add(C21, C13, C13, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16                 | m3+z5+z8         |
        // | z6               | z3+m15+z2        | m26              | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("z6", C11);

        // C33 = m4 - z7 - z8 + m26
        GPU_add(C21, C33, C33, lda, lda, ldc, XA4, YA4, 1.0, 1.0);  // m26 + m4
        GPU_add(C33, C31, C33, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // - z7
        GPU_add(C33, C34, C33, lda, lda, ldc, XA4, YA4, 1.0, -1.0);  // - z8
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              | m3+z5+z8         |
        // | z6               | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | m14+z4+m16       | z8               |                  |

        // print_matrix_4x4("C33", C11);

        // C34 = m3 + z5 + z8 + m25
        // m25 = (X9 + X2 + X10) @ X14.T
        GPU_add(X9, X2, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C11, X10, C11, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_ABt(C11, X14, C34, lda, lda, ldc, XA4, XC4, XC4, YA4, YC4, YC4, 1.0, 0.0);
        GPU_add(C34, C42, C34, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m4               | z7               | m10+z2           |
        // | ---------------- | ---------------- | m16              |                  |
        // | z6               | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | m14+z4+m16       | ---------------- |                  |

        // z6 +> C24 and C41
        GPU_add(C24, C13, C24, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        GPU_add(C41, C13, C41, lda, lda, ldc, XA4, YA4, 1.0, -1.0);
        // |                  | m4               | z7               | m10+z2-z6        |
        // | ---------------- | ---------------- | m16              |                  |
        // |                  | z3+m15+z2        | ---------------- | w1               |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("z6 to C24 and C41", C11);

        // C13 = z2 + z3 + m15 + m16
        GPU_add(C23, C32, C13, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  | m4               | z7               | m10+z2-z6        |
        // | ---------------- | ---------------- | m16              |                  |
        // | ---------------- |                  | ---------------- | w1               |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("C13", C11);

        // C23 = z2 - z6 + z7 + m10
        GPU_add(C41, C31, C23, lda, lda, ldc, XA4, YA4, 1.0, 1.0);
        // |                  |                  |                  |                  |
        // | ---------------- | ---------------- |                  |                  |
        // | ---------------- | ---------------- | ---------------- |                  |
        // | ---------------- | ---------------- | ---------------- |                  |

        // print_matrix_4x4("C23", C11);

        // Recursively calculate C11
        rtxx(X1, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X2, C31, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);

        GPU_add(C21, C31, C11, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);

        rtxx(X3, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);

        rtxx(X4, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C11, C11, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);

        // Recursively calculate C44
        rtxx(X13, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        rtxx(X14, C31, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C31, C44, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);

        rtxx(X15, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);

        rtxx(X16, C21, lda, ldc, XA4, XC4, YA4, YC4, depth - 1);
        GPU_add(C21, C44, C44, ldc, ldc, ldc, XA4, YA4, 1.0, 1.0);
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

    // Create raw pointers for the remaining parts
    Float *a12 = A + nxa;           // pxa x nya starting at (nxa, 0)
    Float *a21 = A + nya * lda;     // nxa x pya starting at (0, nya)
    Float *c21 = C + nyc * ldc;     // nxc x pyc starting at (0, nyc)
    Float *c11 = C;                 // nxc x nyc starting at (0, 0)

    // Final matrix operations
    GPU_ABt(a12, A, c21, lda, lda, ldc, pxa, XA, nxc, nya, YA, pyc, 1.0, 0.0);        // (c21 c22) = (a12 a22)t * A
    GPU_ABt(a21, a21, c11, lda, lda, ldc, nxa, nxa, nxc, pya, pya, nyc, 1.0, 1.0);  // C11 = a21t * a21 + C11
}
