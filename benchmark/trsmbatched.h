#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(seconds, operations) ((1e-9 * 2 * operations) / (seconds))

/**
 * Class TrsmBatched contains TrsmBatched API which solves an array of triangular linear systems with multiple right-hand-sides
 * \Operation: A[i] * X[i] = alpha * B[i]
 * \param A[i]: m x m Triangular matrix stored in lower or upper mode,
 * \param B[i]: m x n general matrix
 * \param X[i]: m x n Matrix to be calculated
 * \param alpha - scalar
 */
template<class T>
class TrsmBatched {
  public:
    /**
     * TrsmBatched constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    TrsmBatched(int A_row, int A_col, int B_row, int B_col, int batch_count, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * TrsmBatchedAPICall function - To allocate Host and Device memory,
          sets up matrices and calls TrsmBatched API based on the mode passed
     */
    int TrsmBatchedApiCall();

  private:
    int A_row, A_col, B_row, B_col, batch_count;
    char mode;
    T **HostMatrixA;
    T **HostMatrixB;
    T **HostPtrToDeviceMatA;
    T **HostPtrToDeviceMatB;
    T **DeviceMatrixA;
    T **DeviceMatrixB;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaEvent_t start, stop;
};
