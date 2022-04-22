%%writefile cublas_DotEx_test.h
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

/**
 * Class DotEx contains DotEx API which performs DotEx product between 2 vectors : 
 * DotEx_product = X.Y
 * \param X - Vector of size n
 * \param Y - Vector of size n
 * \param DotEx_product - output
 */
template<class T>
class DotEx {
  public:
    /**
     * DotEx constructor - To initialize the class varibles using initializer list,
     * sets up the API mode and length of vector
     */
    DotEx(int vector_length, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * DotExApiCall function - To allocate Host and Device memory,
          sets up vector and calls DotEx API based on the mode passed
     */
    int DotExApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T *DotEx_product;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
