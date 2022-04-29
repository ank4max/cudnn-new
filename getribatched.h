%%writefile cublas_GetriBatched_test.h
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
 * Class GetriBatched contains GetriBatched API which performs the inversion of matrices A[i]
 * \Operation: Carray[i] = inv(A[i])
 * \param A[i] - n x n general matrix,
 * \param C[i] - n x n general matrix
 */
template<class T>
class GetriBatched {
  public:
    /**
     * GetriBatched constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    GetriBatched(int A_row, int A_col, int C_row, int C_col, int batch_count, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GetriBatchedAPICall function - To allocate Host and Device memory,
          sets up matrices and calls GetriBatched API based on the mode passed
     */
    int GetriBatchedApiCall();

  private:
    int A_row, A_col, C_row, C_col, batch_count;
    char mode;
    T **HostMatrixA;  
    T **HostPtrToDeviceMatA;
    T **DeviceMatrixA;
    T **HostMatrixC;  
    T **HostPtrToDeviceMatC;
    T **DeviceMatrixC;
    int *HostPivotArray;
    int *DevicePivotArray;
    int *HostInfoArray;
    int *DeviceInfoArray;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
