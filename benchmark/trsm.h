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
 * Class Trsm contains Trsm API which Solves Triangular linear system with multiple right-hand-sides : A * X = alpha * B
 * \param A - m x m triangular matrix in lower mode ,
 * \param B - m x n general matrix
 * \param X - m x n matrix to be calculated
 * \param alpha - scalar
 */
template<class T>
class Trsm {
  public:
    /**
     * Trsm constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha and dimension of matrices
     */
    Trsm(int A_row, int A_col, int B_row, int B_col, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * TrsmAPICall function - To allocate Host and Device memory,
          sets up matrices and calls Trsm API based on the mode passed
     */
    int TrsmApiCall();

  private:
    int A_row, A_col, B_row, B_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixB;
    T *DeviceMatrixA;
    T *DeviceMatrixB;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaEvent_t start, stop;
};
