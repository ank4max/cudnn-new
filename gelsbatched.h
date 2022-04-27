%%writefile cublas_Gelsbatched_test.h
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
 * Class Gelsbatched contains Gelsbatched API which performs matrix-matrix multiplication of a batch of matrices
 * \Operation : minimize  || Carray[i] - Aarray[i]*Xarray[i] ||
 * \param A[i] - m x k general matrix,
 * \param C[i] - m x n general matrix
 */
template<class T>
class Gelsbatched {
  public:
    /**
     * Gelsbatched constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, and dimension of matrices
     */
    Gelsbatched(int A_row, int A_col, int C_row, int C_col, int batch_count, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GelsbatchedAPICall function - To allocate Host and Device memory,
          sets up matrices and calls Gelsbatched API based on the mode passed
     */
    int GelsbatchedApiCall();

  private:
    int A_row, A_col, C_row, C_col, batch_count;
    char mode;
    T **HostMatrixA;  
    T **HostPtrToDeviceMatA;
    T **DeviceMatrixA;
    T **HostMatrixC;  
    T **HostPtrToDeviceMatC;
    T **DeviceMatrixC;
    int *HostInfoArray;
    int *DeviceInfoArray;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
