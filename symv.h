%%writefile symv.h
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
 * Class Symv contains Symv API which performs the symmetric matrix-vector multiplication : Y = alpha * A * X + beta * Y
 * \param A - n x n symmetric matrix,
 * \param X - vector of length n
 * \param Y - vector of length n
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class Symv {
  public:
    /**
     * Symv constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta  and dimension of matrix and vectors
     */
    Symv(int A_row, int A_col, int vector_length, T alpha, T beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * SymvAPICall function - To allocate Host and Device memory,
          sets up matrix and vector and calls Symv API based on the mode passed
     */
    int SymvApiCall();

  private:
    int A_row, A_col;
    int vector_length;
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
