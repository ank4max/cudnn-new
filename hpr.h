%%writefile hpr.h
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
 * Class Hpr contains Hpr API which performs the packed Hermitian rank-1 update : A = alpha * X * X ^ H + A
 * \param A - n x n Hermitian packed matrix ,
 * \param X - vector of length n
 * \param alpha - scalar
 */
template<class T>
class Hpr {
  public:
    /**
     * Hpr constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, and dimension of matrix and vector
     */
    Hpr(int A_row, int A_col, int vector_length, double alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * HprAPICall function - To allocate Host and Device memory,
          sets up matrix and vector and calls Hpr API based on the mode passed
     */
    int HprApiCall();

  private:
    int A_row, A_col, vector_length ;
    char mode;
    T *HostMatrixA;
    T *HostVectorX;
    T *DeviceMatrixA;
    T *DeviceVectorX;
    double alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
