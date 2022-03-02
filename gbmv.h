%%writefile gbmv.h
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
 * Class Gbmv contains Gbmv API which performs matrix-matrix multiplication : C = alpha * A * B + beta * C
 * \param A - m x k general matrix,
 * \param B - k x n general matrix
 * \param C - m x n general matrix
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class Gbmv {
  public:
    /**
     * Gbmv constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    Gbmv(int A_row, int A_col, int x_size, int y_size,  int super_diagonals, int sub_diagonals, T alpha, T beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GbmvAPICall function - To allocate Host and Device memory,
          sets up matrices and calls Gbmv API based on the mode passed
     */
    int GbmvApiCall();

  private:
    int A_row, A_col;
    int x_size, y_size,  super_diagonals, sub_diagonals;
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
