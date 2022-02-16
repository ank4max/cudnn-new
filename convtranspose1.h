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
 * Class Convolution contains ConvolutionBackward API which performs Backward Convolution operation on Input Image
   which is same as transpose covolution: 
 */
class Convolution {
  public:
    /**
     * Convolution constructor - To initialize the class varibles using initializer list
     */
    Convolution(int batch, int channel, int height, int width, int filter_batch,
                int filter_channel, int filter_height, int filter_width, int padding, 
                int stride, int dilation);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * ConvolutionBackwardApiCall which performs Convolution Transpose operation on input image
     */
    int ConvolutionBackwardApiCall();
    
  private:
    int batch, channel, height, width;
    int output_batch, output_channel, output_height, output_width;
    int filter_batch, filter_channel, filter_height, filter_width;
    int padding, stride, dilation;
    int padding_height;
    int padding_width;
    int stride_height;
    int stride_width;
    int dilation_height;
    int dilation_width;
    float alpha = 1.0;
    float beta = 0.0;
    float *workspace_data;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *HostFilterTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    float *DeviceFilterTensor;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t convolution_desc;
    cudnnConvolutionBwdDataAlgo_t convolution_algo;
     
};
 
