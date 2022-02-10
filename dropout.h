%%writefile convolution.h
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>
#include <vector>
#include <iomanip>

/* 
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 * finally dividing it by latency to get required throughput 
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

/**
 * Class Dropout contains DropoutForward API which performs Forward Convolution operation on Input Image : 
 */
class Dropout {
  public:
    /**
     * Dropout constructor - To initialize the class varibles using initializer list
     */
    Dropout(int batch, int channel, int height, int width, float drop_rate);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    int FreeMemory();

    /**
     * DropoutForwardApiCall which performs Dropout operation on input image
     */
    int DropoutForwardApiCall();
    
  private:
    int batch, channel, height, width, Dropout;
    float* d_dropout_out{nullptr};
	  float* d_dx_dropout{nullptr};
    size_t dropout_state_size;
	  size_t dropout_reserve_size;
    void* states;
	  void* dropout_reserve_space;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    cudnnDropoutDescriptor_t dropout_descriptor;
    cudnnTensorDescriptor_t dropout_in_out_descriptor;
    
};
