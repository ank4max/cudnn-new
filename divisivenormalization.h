%%writefile div.h
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
 * Class DivisiveNormalizationForward contains DivisiveNormalizationForward API which performs the forward spatial DivisiveNormalization layer computation.
 */
class DivisiveNormalizationForward {
  public:
    /**
     * DivisiveNormalizationForward constructor - To initialize the class varibles using initializer list
     */
    DivisiveNormalizationForward(int batch, int channel, int height, int width);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * DivisiveNormalizationForwardApiCall which performs the forward spatial DivisiveNormalization layer computation.
     */
    int DivisiveNormalizationForwardApiCall();

  private:
    int batch, channel, height, width;
    float alpha;
    float beta;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor; 
    float *means, *temp, *temp2;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnDivNormMode_t divisivenorm_mode;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnLRNDescriptor_t DivisiveNorm_descriptor;
};
