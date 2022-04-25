%%writefile cublas_ScalEx_test.h
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
 * Class ScalEx contains ScalEx API which Scales the vector x by the ScalExar α and overwrites it with the result : X = alpha * X
 * \param X - vector of length n
 * \param alpha - Scalar
 */
template<class T>
class ScalEx {
  public:
    /**
     * ScalEx constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, alpha and dimension of vector
     */
    ScalEx(int vector_length, T alpha, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * ScalExAPICall function - To allocate Host and Device memory,
          sets up vector and calls ScalEx API based on the mode passed
     */
    int ScalExApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *DeviceVectorX;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
