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
 * Class Trmm contains Trmm API which performs Triangular matrix - matrix multiplication : C = alpha * A * B
 * \param A - m x m triangular matrix in lower mode ,
 * \param B - m x n general matrix
 * \param C - m x n general matrix
 * \param alpha - scalar
 */
template<class T>
class Trmm {
  public:
    /**
     * Trmm constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha and dimension of matrices
     */
    Trmm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * TrmmAPICall function - To allocate Host and Device memory,
          sets up matrices and calls Trmm API based on the mode passed
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
    cudaEvent_t start, stop;
};
