%%writefile cublas_MatrixTransform_test.h
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>
#include "cublas_utility.h"

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start))

/**
 * Class MatrixTransform contains MatrixTransform API which computes the matrix transformation operation on the input matrices A and B, to 
      produce the output matrix C : C = alpha * transform(A) + beta * transform(B) 
 * \param A - m x k general matrix,
 * \param B - k x n general matrix
 * \param C - m x n general matrix
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class MatrixTransform {
  public:
    /**
     * MatrixTransform constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    MatrixTransform(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * MatrixTransformAPICall function - To allocate Host and Device memory,
          sets up matrices and calls MatrixTransform API based on the mode passed
     */
    int MatrixTransformApiCall();

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
    cublasLtMatrixTransformDesc_t transformDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasLtHandle_t LtHandle;
    clock_t clk_start, clk_end;
};
