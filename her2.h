%%writefile her2.h
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
 * Class Her2 contains Her2 API which performs the Hermitian rank-2 update : A = alpha * X * Y ^ H + alpha(Imaginary part) * Y * X ^ H + A 
 * \param A - n x n hermitian matrix ,
 * \param X - vector of length n
 * \param Y - vector of length n
 * \param alpha - scalar
 */
template<class T>
class Her2 {
  public:
    /**
     * Her2 constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, and dimension of matrix and vectors
     */
    Her2(int A_row, int A_col, int vector_length, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * Her2APICall function - To allocate Host and Device memory,
          sets up matrix and vectors and calls Her2 API based on the mode passed
     */
    int Her2ApiCall();

  private:
    int A_row, A_col, vector_length ;
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
