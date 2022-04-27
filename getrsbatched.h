%%writefile cublas_GetrsBatched_test.h
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start))

/**
 * Class GetrsBatched contains GetrsBatched API which performs matrix-matrix multiplication of a batch of matrices
 * \Operation: C[i] = alpha * A[i] * B[i] + beta * C[i]
 * \param A[i] - m x k general matrix,
 * \param B[i] - k x n general matrix
 * \param C[i] - m x n general matrix
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class GetrsBatched {
  public:
    /**
     * GetrsBatched constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    GetrsBatched(int A_row, int A_col, int B_row, int B_col, int batch_count, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GetrsBatchedAPICall function - To allocate Host and Device memory,
          sets up matrices and calls GetrsBatched API based on the mode passed
     */
    int GetrsBatchedApiCall();

  private:
    int A_row, A_col, B_row, B_col, batch_count;
    char mode;
    T **HostMatrixA;  
    T **HostPtrToDeviceMatA;
    T **DeviceMatrixA;
    T **HostMatrixB;  
    T **HostPtrToDeviceMatB;
    T **DeviceMatrixB;
    int *HostdevIpiv;
    int *DevicedevIpiv;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
