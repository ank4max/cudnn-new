%%writefile cublas_GetrfBatched_test.h
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
 * Class GetrfBatched contains GetrfBatched API which performs the LU factorization of each Aarray[i] :\n
 * \Operation: P * Aarray[i] = L * U
 * \param A[i] - n x n general matrix,
 */
template<class T>
class GetrfBatched {
  public:
    /**
     * GetrfBatched constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    GetrfBatched(int A_row, int A_col, int batch_count, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GetrfBatchedAPICall function - To allocate Host and Device memory,
          sets up matrices and calls GetrfBatched API based on the mode passed
     */
    int GetrfBatchedApiCall();

  private:
    int A_row, A_col, batch_count;
    char mode;
    T **HostMatrixA;  
    T **HostPtrToDeviceMatA;
    T **DeviceMatrixA;
    int *HostPivotArray;
    int *DevicePivotArray;
    int *HostInfoArray;
    int *DeviceInfoArray;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
