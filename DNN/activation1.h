%%writefile activation.h
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
 * Class Activation contains Activation API which performs activation operation on Input Image based on mode :
 */
class Activation {
  public:
    /**
     * Activation constructor - To initialize the class varibles using initializer list
     */
    Activation(int batch, int channel, int height, int width, char* activation_mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * ActivationApiCall which performs activation operation on input image
     */
    int ActivationApiCall();

  private:
    int batch, channel, height, width;
    std::string activation_mode;
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
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t prop;
    cudnnActivationDescriptor_t activation_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;

};
