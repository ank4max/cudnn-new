%%writefile dgmm.h
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
 * Class Dgmm contains Dgmm API which performs matrix-matrix multiplication : C = A * diag(X)
 * \param A - m x n general matrix,
 * \param X - vector of length n
 * \param C - m x n general matrix,
 */
template<class T>
class Dgmm {
  public:
    /**
     * Dgmm constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of matrix and vectors
     */
    Dgmm(int A_row, int A_col, int C_row, int C_col, int vector_length, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * DgmmAPICall function - To allocate Host and Device memory,
          sets up matrix and vectors and calls Dgmm API based on the mode passed
     */
    int DgmmApiCall();

  private:
    int A_row, A_col, C_row, C_col;
    int vector_length;
    char mode;
    T *HostMatrixA;
    T *HostVectorX;
    T *HostMatrixC;
    T *DeviceMatrixA;
    T *DeviceVectorX;
    T *DeviceMatrixC;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
