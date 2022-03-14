%%writefile tbsv.h
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
 * Class Tbsv contains Tbsv API which solves the triangular banded linear system with a single right-hand-side : op(A) * X = b
 * \param A - n x n triangular banded matrix,
 * \param X - vector of length n
 * \param b - vector of length n
 */
template<class T>
class Tbsv {
  public:
    /**
     * Tbsv constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of matrix and vector
     */
    Tbsv(int A_row, int A_col, int vector_length, int sub_diagonals, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * TbsvAPICall function - To allocate Host and Device memory,
          sets up matrix and vector and calls Tbsv API based on the mode passed
     */
    int TbsvApiCall();

  private:
    int A_row, A_col;
    int vector_length, sub_diagonals;
    char mode;
    T *HostMatrixA;
    T *HostVectorX;
    T *DeviceMatrixA;
    T *DeviceVectorX;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
