%%writefile max.h
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
 * Class Amax contains Amax API which finds the (smallest) index of the element of the maximum magnitude
 * \param X - n x 1 general vector
 */
template<class T>
class Amax {
  public:
    /**
     * Amax constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, and dimension of vector
     */
    Amax(int x_size, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * AmaxAPICall function - To allocate Host and Device memory,
          sets up vector and calls Amax API based on the mode passed
     */
    int AmaxApiCall();

  private:
    int x_size;
    char mode;
    T *HostVectorX;
    T *DeviceVectorX;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
