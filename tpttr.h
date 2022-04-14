%%writefile tpr.h
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
 * Class Tpttr contains Tpttr API which performs the conversion from the triangular packed format to the triangular format : AP -> A
 * \param AP - n x (n + 1)/2 triangular packed matrix,
 * \param A - n  x n triangular matrix,
 */
template<class T>
class Tpttr {
  public:
    /**
     * Tpttr constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of matrix and vector
     */
    Tpttr(int A_row, int A_col, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * TpttrAPICall function - To allocate Host and Device memory,
     * sets up matrix and vector and calls Tpttr API based on the mode passed
     */
    int TpttrApiCall();

  private:
    int A_row, A_col;
    char mode;
    T *HostMatrixAP;
    T *HostMatrixA;
    T *DeviceMatrixAP;
    T *DeviceMatrixA;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
