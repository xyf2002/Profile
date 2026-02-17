#include "cublas_wrapper.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string.h>

#define MAX_ERROR_MSG 256

/* Global state */
static cublasHandle_t g_handle = NULL;
static char g_error_msg[MAX_ERROR_MSG] = {0};

/* Helper macro to check CUDA errors */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            snprintf(g_error_msg, MAX_ERROR_MSG, \
                     "CUDA Error: %s", cudaGetErrorString(err)); \
            return 1; \
        } \
    } while(0)

/* Helper macro to check cuBLAS errors */
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            snprintf(g_error_msg, MAX_ERROR_MSG, \
                     "cuBLAS Error: status code %d", status); \
            return 1; \
        } \
    } while(0)

int cublas_init(void) {
    if (g_handle != NULL) {
        /* Already initialized */
        return 0;
    }
    
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        snprintf(g_error_msg, MAX_ERROR_MSG,
                 "Failed to create cuBLAS handle: status code %d", status);
        g_handle = NULL;
        return 1;
    }
    
    return 0;
}

void cublas_cleanup(void) {
    if (g_handle != NULL) {
        cublasDestroy(g_handle);
        g_handle = NULL;
    }
}

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
) {
    if (g_handle == NULL) {
        snprintf(g_error_msg, MAX_ERROR_MSG, "cuBLAS not initialized");
        return 1;
    }
    
    if (A == NULL || B == NULL || C == NULL) {
        snprintf(g_error_msg, MAX_ERROR_MSG, "NULL pointer passed to cublas_sgemm");
        return 1;
    }
    
    /* Convert transpose flags */
    cublasOperation_t transaOp = (transa == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transbOp = (transb == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    
    /* Calculate device memory requirements */
    size_t size_A = (size_t)lda * (transa == 0 ? k : m) * sizeof(float);
    size_t size_B = (size_t)ldb * (transb == 0 ? n : k) * sizeof(float);
    size_t size_C = (size_t)ldc * n * sizeof(float);
    
    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));
    
    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
    
    /* Perform SGEMM on GPU */
    CUBLAS_CHECK(cublasSgemm(
        g_handle,
        transaOp,
        transbOp,
        m,
        n,
        k,
        &alpha,
        d_A,
        lda,
        d_B,
        ldb,
        &beta,
        d_C,
        ldc
    ));
    
    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    /* Cleanup device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

const char* cublas_get_error(void) {
    return g_error_msg[0] ? g_error_msg : "No error";
}
