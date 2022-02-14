#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>

/* 
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 * finally dividing it by latency to get required throughput 
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

/**
 * Class Pooling contains PoolingForward API which performs Max-pool operation on Input Image : 
 */
class Pooling {
  public:
    /**
     * Pooling constructor - To initialize the class varibles using initializer list
     */
    Pooling(int batch, int channel, int height, int width);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * PoolingForwardApiCall which performs Max-pool operation on input image
     */
    int PoolingForwardApiCall();
    
  private:
    int batch, channel, height, width;
    float alpha = 1.0;
    float beta = 0.0;
    float *input;
    float *output;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
};

