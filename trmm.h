#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/**
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 * finally dividing it by latency to get required throughput 
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

/**
 * Template class Trmm is defined having matrices ,their dimensions,
      mode and scalars quantity declared as private members
 * Cublas handle, cuda status and cublas status are also declared as private members
 * Clock varibles clk_start and clk_end are to compute throughput and latency
 */
template<class T>
class Trmm {
  public:
    /**
     * Trmm constructor - To initialize the class varibles using initializer list, 
     * sets up the dimension of matrices, alpha and API mode
     * This API performs triangular matrix - matrix multiplication : DeviceMatrixC = alpha * DeviceMatrixA * DeviceMatrixB 
     * DeviceMatrixA - m x m triangular matrix in lower mode ,
     * DeviceMatrixB, DeviceMatrixC - m x n general matrices and alpha - scalar  
     */
    Trmm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, char mode);
    
    /** 
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();
    
    /**
     * TrmmAPICall function - To allocate host and device memory,
          sets up matrices and calls trmm API based on the mode passed
     */
    int TrmmApiCall();

  private:
    int A_row, A_col, B_row, B_col, C_row, C_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixB;
    T *HostMatrixC;
    T *DeviceMatrixA;
    T *DeviceMatrixB;
    T *DeviceMatrixC;
    T alpha;
    T beta;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
