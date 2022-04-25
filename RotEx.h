#include <iostream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start))

//! To convert Degree into Radian  - pi/180
#define DEG_TO_RADIAN(degree)  degree * 0.01745329251

/**
 * Class RotEx contains RotEx API which applies Givens Rotation matrix (i.e., Rotation in the x,y plane counter-clockwise
 * by angle defined by cos(alpha) = c, sin(alpha) = s)
 * \param X - vector of length n
 * \param Y - vector of length n
 * \param sine - scalar
 * \param cosine - scalar
 */
template<class T>
class RotEx {
  public:
    /**
     * RotEx constructor - To initialize the class varibles using initializer list,
     * sets up the API mode, sine, cosine and dimension of vectors
     */
    RotEx(int vector_length, T sine, T cosine, char mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * RotExAPICall function - To allocate Host and Device memory,
          sets up vectors and calls RotEx API based on the mode passed
     */
    int RotExApiCall();

  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T sine;
    T cosine;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;
};
