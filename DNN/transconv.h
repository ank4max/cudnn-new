%%writefile transconv.h
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
 * Class ConvolutionTranspose contains ConvolutionTranspose API which performs Transpose Convolution operation on Input Image :
 */
class ConvolutionTranspose {
  public:
    /**
     * ConvolutionTranspose constructor - To initialize the class varibles using initializer list
     */
    ConvolutionTranspose(int batch, int channel, int height, int width, int filter_batch,
                         int filter_channel, int filter_height, int filter_width, int padding,
                         int stride, int dilation, char *bwd_preference);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * ConvolutionTransposeApiCall which performs Convolution operation on input image
     */
    int ConvolutionTransposeApiCall();

  private:
    int batch, channel, height, width;
    std::string bwd_preference;
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
    float zero;
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
    cudnnConvolutionBwdDataPreference_t data_preference;
};
