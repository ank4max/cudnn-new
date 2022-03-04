%%writefile Swap.h
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
 * Class Swap contains Swap API which interchanges the elements of vector x and y : Y ⇔ X
 * \param X - vector of length n
 * \param Y - vector of length n
 */
template<class T>
class Swap {
  public:
    /**
     * Swap constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of vectors
     */
    Swap(int vector_length, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * SwapAPICall function - To allocate Host and Device memory,
          sets up vectors and calls Swap API based on the mode passed
     */
    int SwapApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
