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
 * Class BatchNormalization contains BatchNormalization API which performs batchnormal forward training  operation on Input Image based on mode :
 */
class BatchNormalizationForward {
  public:
    /**
     * BatchNormalization constructor - To initialize the class varibles using initializer list
     */
    BatchNormalizationForward(int batch, int channel, int height, int width, char *batchnorm_mode);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * BatchNormalizationApiCall which performs batchnormal forward training operation on input image
     */
    int BatchNormalizationForwardApiCall();

  private:
    int batch, channel, height, width;
    std::string batchnorm_mode;
    float alpha;
    float beta;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    float *scale, *offset;
    float *device_scale, *device_offset;
    float *running_mean, *running_var;
    float *device_running_mean, *device_running_var;
    float *device_saved_mean, *device_saved_inv_var;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
    cudnnActivationDescriptor_t activation_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t mean_descriptor;
};
