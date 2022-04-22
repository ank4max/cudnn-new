%%writefile cublas_Nrm2Ex_test.h
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
 * Class Nrm2Ex contains Nrm2Ex API which computes the Euclidean norm of the vector x
 * \param X - vector of length n
 */
template<class T>
class Nrm2Ex {
  public:
    /**
     * Nrm2Ex constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and dimension of vector
     */
    Nrm2Ex(int vector_length, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * Nrm2ExAPICall function - To allocate Host and Device memory,
          sets up vector and calls Nrm2Ex API based on the mode passed
     */
    int Nrm2ExApiCall();

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
