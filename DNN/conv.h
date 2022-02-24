%%writefile connv.h
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
 * Class ConvolutionForward contains ConvolutionForward API which performs Convolution forward operation on Input Image : 
 */
class ConvolutionForward {
  public:
    /**
     * ConvolutionForward constructor - To initialize the class varibles using initializer list
     */
    ConvolutionForward(int batch, int channel, int height, int width, int filter_batch,
                       int filter_channel, int filter_height, int filter_width, int padding, 
                       int stride, int dilation, char *mode, char *preference);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * ConvolutionForwardApiCall which performs Convolution forward operation on input image
     */
    int ConvolutionForwardApiCall();
    
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
    float alpha;
    float beta;
    std::string mode, preference;
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
    cudnnConvolutionFwdAlgo_t convolution_algo;
    cudnnConvolutionMode_t convolution_mode;
    cudnnConvolutionFwdPreference_t data_preference; 
};
