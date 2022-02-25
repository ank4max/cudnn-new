%%writefile batchnorm.h
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
 * Class BatchNormalizationBackward contains BatchNormalizationBackward API which performs batchnormal forward training  operation on Input Image based on mode :
 */
class BatchNormalizationBackward {
  public:
    /**
     * BatchNormalizationBackward constructor - To initialize the class varibles using initializer list
     */
    BatchNormalizationBackward(int batch, int channel, int height, int width, char *mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * BatchNormalizationBackwardApiCall which performs batchnormal backward training operation on input image
     */
    int BatchNormalizationBackwardApiCall();

  private:
    int batch, channel, height, width;
    std::string mode;
    float alpha;
    float beta;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    float *scale;
    float *device_scale;
    float *running_mean, *running_var;
    float *device_running_mean, *device_running_var;
    float *device_saved_mean, *device_saved_inv_var;
    float *result;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnBatchNormMode_t bn_mode;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t mean_descriptor;
};
