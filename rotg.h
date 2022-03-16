%%writefile rotg.h
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
 * Class Rotg contains Rotg API which constructs the Givens rotation matrix :
 * \param a - scalar
 * \param b - scalar
 * \param sine - scalar
 * \param cosine - scalar
 */
template<class T>
class Rotg {
  public:
    /**
     * Rotg constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and scalars
     */
    Rotg(T scalar_a, T scalar_b, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * RotgAPICall function - To allocate Host and Device memory,
          sets up scalars and calls Rotg API based on the mode passed
     */
    int RotgApiCall();

  private:
    char mode;
    T scalar_a, scalar_b;
    T sine;
    double cosine;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
