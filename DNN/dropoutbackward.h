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
 * Class DropoutBackward contains DropoutBackward API which performs Dropout Backward operation on Input Image :
 */
class DropoutBackward {
  public:
    /**
     * DropoutBackward constructor - To initialize the class varibles using initializer list
     */
    DropoutBackward(int batch, int channel, int height, int width, float drop_rate);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * DropoutBackwardApiCall which performs Dropout Backward operation on input image
     */
    int DropoutBackwardApiCall();

  private:
    int batch, channel, height, width;
    float dropout_rate;
    size_t dropout_state_size;
    size_t dropout_reserve_size;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    void *states;
    void *dropout_reserve_space;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnDropoutDescriptor_t dropout_descriptor;
    cudnnTensorDescriptor_t input_desc;
};
