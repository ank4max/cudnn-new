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
 * Class BatchNormalizationForwardTrainingEx contains BatchNormalizationForwardTrainingEx API which performs the forward batch normalization layer computation.
 */
class BatchNormalizationForwardTrainingEx {
  public:
    /**
     * BatchNormalizationForwardTrainingEx constructor - To initialize the class varibles using initializer list
     */
    BatchNormalizationForwardTrainingEx(int batch, int channel, int height, int width, char *batchnorm_mode, char *activate_mode, char *norm_ops);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * BatchNormalizationForwardTrainingExApiCall which performs forward batch normalization layer computation.
     */
    int BatchNormalizationForwardTrainingExApiCall();

  private:
    int batch, channel, height, width;
    std::string batchnorm_mode, activate_mode, norm_ops;
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
    float *workspace_data;
    float *reserve_data;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnBatchNormMode_t bn_mode;
    cudnnActivationMode_t activation_mode;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t mean_descriptor;
    cudnnNanPropagation_t propagation = CUDNN_NOT_PROPAGATE_NAN;
    cudnnActivationDescriptor_t activation_desc;
    cudnnBatchNormOps_t bn_ops;

};
