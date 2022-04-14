%%writefile GemmEx.h
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
 * Class GemmEx contains GemmEx API which performs matrix-matrix multiplication : C = alpha * A * B + beta * C
 * This API is an extension of cublas<t>gemm
 * \param A - m x k general matrix,
 * \param B - k x n general matrix
 * \param C - m x n general matrix
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class GemmEx {
  public:
    /**
     * GemmEx constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    GemmEx(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode, char algo);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * GemmExAPICall function - To allocate Host and Device memory,
          sets up matrices and calls GemmEx API based on the mode passed
     */
    int GemmExApiCall();

  private:
    int A_row, A_col, B_row, B_col, C_row, C_col;
    char mode;
    char algo;
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
    cublasGemmAlgo_t cublas_algo;
};
