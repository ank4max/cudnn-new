%%writefile nrm2.h
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
 * Class Nrm2 contains Nrm2 API which computes the Euclidean norm of the vector x
 * \param X - n x 1 general vector
 */
template<class T>
class Nrm2 {
  public:
    /**
     * Nrm2 constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of vector
     */
    Nrm2(int vector_length, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * Nrm2APICall function - To allocate Host and Device memory,
          sets up vector and calls Nrm2 API based on the mode passed
     */
    int Nrm2ApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *DeviceVectorX;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
