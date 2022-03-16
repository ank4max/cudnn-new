%%writefile rotmg.h
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
 * Class Rotmg contains Rotmg API which applies the modified Givens transformation : 
 * The result is that zeros out the second entry of a 2×1 vector(d1 * 1/2 * x1, d2 * 1/2 * y1) * T.
 * \param Param - 2 x 2 matrix,
 * \param d1 - scalar
 * \param d2 - scalar
 * \param x1 - scalar
 * \param y1 - scalar
 */
template<class T>
class Rotmg {
  public:
    /**
     * Rotmg constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, param matrix and scalars
     */
    Rotmg(T scalar_d1, T scalar_d2, T scalar_x1, T scalar_y1, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * RotmgAPICall function - To allocate Host and Device memory,
          sets up param matrix ,scalars and calls Rotmg API based on the mode passed
     */
    int RotmgApiCall();

  private:
    char mode;
    T *HostMatrixParam;
    T scalar_d1, scalar_d2, scalar_x1, scalar_y1;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
