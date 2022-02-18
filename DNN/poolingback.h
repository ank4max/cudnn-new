%%writefile poolingbackward.h
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>
#include <iomanip>

/*
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 * finally dividing it by latency to get required throughput
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start))

/**
 * Class PoolingBackward contains PoolingBackward API which performs Max-pool operation on Input Image :
 */
class PoolingBackward {
  public:
    /**
     * PoolingBackward constructor - To initialize the class varibles using initializer list
     */
    PoolingBackward(int batch, int channel, int height, int width, int window, int padding, int stride, char* pooling_mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * PoolingBackwardApiCall which performs Max-pool operation on input image
     */
    int PoolingBackwardApiCall();

  private:
    int batch, channel, height, width, window, padding, stride;
    std::string pooling_mode;
    float alpha = 1.0;
    float beta = 0.0;
    int vertical_padding;
    int horizontal_padding;
    int window_height;
    int window_width;
    int vertical_stride;
    int horizontal_stride;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    cudnnPoolingMode_t mode;
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
};
