%%writefile Rot.h
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
 * Class Rot contains Rot API which applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise 
 * by angle defined by cos(alpha) = c, sin(alpha) = s)
 * \param X - vector of length n
 * \param Y - vector of length n
 * \param sine - scalar
 * \param cosine - scalar
 */
template<class T, class C>
class Rot {
  public:
    /**
     * Rot constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, sine, cosine and dimension of vectors
     */
    Rot(int vector_length, C sine, double cosine, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * RotAPICall function - To allocate Host and Device memory,
          sets up vectors and calls Rot API based on the mode passed
     */
    int RotApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    C sine;
    double cosine;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
