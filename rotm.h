%%writefile rotm.h
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
 * Class Rotm contains Rotm API which applies the modified Givens transformation : 
 * \param Param - 2 x 2 matrix,
 * \param X - vector of lenght n
 * \param Y - vector of length n
 */
template<class T>
class Rotm {
  public:
    /**
     * Rotm constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha, beta and dimension of matrix and vectors
     */
    Rotm(int vector_length, T param_1, T param_2, T param_3, T param, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * RotmAPICall function - To allocate Host and Device memory,
          sets up matrix and vectors and calls Rotm API based on the mode passed
     */
    int RotmApiCall();

  private:
    int vector_length;
    char mode;
    T *HostMatrixParam;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T param_1, param_2, param_3, param_4;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
