#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublasXt.h"
#include "cublas_utility.h"

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start))

/**
 * Class Herkx contains Herkx API which performs variation of the Hermitian rank- k update : C = alpha * A * B^H + beta * C
 * \param A - n x k general matrix,
 * \param B - n x k general matrix
 * \param C - n x n Hermitian matrix stored in lower or upper mode
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class Herkx {
  public:
    /**
     * Herkx constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    Herkx(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, T alpha, double beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * HerkxAPICall function - To allocate Host and Device memory,
          sets up matrices and calls Herkx API based on the mode passed
     */
    int HerkxApiCall();

  private:
    size_t A_row, A_col, B_row, B_col, C_row, C_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixB;
    T *HostMatrixC;
    T alpha;
    double beta;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasXtHandle_t handle;
    clock_t clk_start, clk_end;
};
