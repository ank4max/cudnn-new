%%writefile ger.h
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
 * Class Ger contains Ger API which performs the rank-1 update: A = alpha * X * Y ^ T + A or A = alpha * X * Y ^ H + A
 * \param A - m x n general matrix,
 * \param X - vector of length n
 * \param Y - vector of length m
 * \param alpha - scalar
 */
template<class T>
class Ger {
  public:
    /**
     * Ger constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, and dimension of matrix and vectors
     */
    Ger(int A_row, int A_col, int X_length, int Y_length, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GerAPICall function - To allocate Host and Device memory,
          sets up matrix and vectors and calls Ger API based on the mode passed
     */
    int GerApiCall();

  private:
    int A_row, A_col;
    int X_length, Y_length;
    char mode;
    T *HostMatrixA;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceMatrixA;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
