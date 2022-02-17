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
 * Class BatchNormalization contains BatchNormalization API which performs batchnormal forward training  operation on Input Image based on mode :
 */
class BatchNormalization {
  public:
    /**
     * BatchNormalization constructor - To initialize the class varibles using initializer list
     */
    BatchNormalization(int batch, int channel, int height, int width);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * BatchNormalizationApiCall which performs batchnormal forward training operation on input image
     */
    int BatchNormalizationApiCall();

  private:
    int batch, channel, height, width;
    float one = 1.0;
    float zero = 0.0;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *scale, *offset, *dscale, *doffset;
    float *running_mean, *running_var;
    float *saved_mean, *saved_inv_var;
    char *reserve_space;
    void *workspace = nullptr;
    size_t workspace_size_bytes = 0;
    size_t reserve_space_size_bytes = 0;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
    cudnnActivationDescriptor_t activation_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t mean_descriptor;

};
