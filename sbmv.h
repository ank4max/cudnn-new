%%writefile sbmv.h
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
 * Class Sbmv contains Sbmv API which performs the symmetric banded matrix-vector multiplication : Y = alpha * A * X + beta * Y 
 * \param A - n x n symmetric banded  matrix,
 * \param X - vector of length n,
 * \param Y - vector of length n,
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class Sbmv {
  public:
    /**
     * Sbmv constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrix and vectors 
     */
    Sbmv(int A_row, int A_col, int vector_length, int sub_diagonals, T alpha, T beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * SbmvAPICall function - To allocate Host and Device memory,
          sets up matrix and vectors and calls Sbmv API based on the mode passed
     */
    int SbmvApiCall();

  private:
    int A_row, A_col;
    int vector_length, sub_diagonals;
    char mode;
    T *HostMatrixA;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceMatrixA;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T alpha;
    T beta;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
