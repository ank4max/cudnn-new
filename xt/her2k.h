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
 * Class Her2k contains Her2k API which performs Hermitian rank-2k update : 
 * C = alpha * A * B^H + alpha * B * A^H  + beta * C
 * \param A - n x k general matrix,
 * \param B - n x k general matrix
 * \param C - n x n Hermitian matrix
 * \param alpha - scalar
 * \param beta - scalar
 */
template<class T>
class Her2k {
  public:
    /**
     * Her2k constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrices
     */
    Her2k(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, double beta, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * Her2kAPICall function - To allocate Host memory,
          sets up matrices and calls Her2k API based on the mode passed
     */
    int Her2kApiCall();
    
  private:
    int A_row, A_col, B_row, B_col, C_row, C_col;
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
