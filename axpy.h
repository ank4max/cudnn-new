%%writefile axpy.h
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
 * Class Axpy contains Axpy API which multiplies the vector x by the scalar α and adds it to the vector y 
 * overwriting the latest vector with the result : Y = alpha * X + Y
 * \param X - vector of length n
 * \param Y - vector of length n
 * \param alpha - scalar
 */
template<class T>
class Axpy {
  public:
    /**
     * Axpy constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha and dimension of vectors
     */
    Axpy(int vector_length, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * AxpyAPICall function - To allocate Host and Device memory,
          sets up vectors and calls Axpy API based on the mode passed
     */
    int AxpyApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
