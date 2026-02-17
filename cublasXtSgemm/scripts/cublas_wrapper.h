#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize cuBLAS context
 * Returns: 0 on success, non-zero on error
 */
int cublas_init(void);

/**
 * Cleanup cuBLAS context
 */
void cublas_cleanup(void);

/**
 * Perform SGEMM operation on GPU
 * Allocates GPU memory, transfers data, performs calculation, and transfers result back
 * 
 * Parameters:
 *   transa, transb: 0 for 'N' (no transpose), 1 for 'T' (transpose)
 *   m, n, k: Matrix dimensions (C[m,n] = alpha*A[m,k]*B[k,n] + beta*C[m,n])
 *   alpha, beta: Scaling factors
 *   A: Input matrix A (host memory), size m*k
 *   B: Input matrix B (host memory), size k*n
 *   C: Output matrix C (host memory), size m*n
 * 
 * Returns: 0 on success, non-zero on error
 */
int cublas_sgemm(
    int transa,
    int transb,
    int m,
    int n,
    int k,
    float alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta,
    float *C,
    int ldc
);

/**
 * Get last error message
 * Returns: pointer to error message string
 */
const char* cublas_get_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CUBLAS_WRAPPER_H */
