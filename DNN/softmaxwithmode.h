%%writefile cudnn_softmaxforward_test.h
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
 * Class SoftmaxForward contains SoftmaxForward API which performs softmax operation on Input Image :
 */
class SoftmaxForward {
  public:
    /**
     * SoftmaxForward constructor - To initialize the class varibles using initializer list
     */
    SoftmaxForward(int batch, int channel, int height, int width, char *mode, char *algo);

    /**
     * FreeMemory function - To free the allocated memory when program is ended or in case of any error
     */
    void FreeMemory();

    /**
     * SoftmaxForwardApiCall which performs softmax operation on input image
     */
    int SoftmaxForwardApiCall();

  private:
    int batch, channel, height, width;
    std::string mode, algo;
    float alpha;
    float beta;
    float *HostInputTensor;
    float *HostOutputTensor;
    float *DeviceInputTensor;
    float *DeviceOutputTensor;
    clock_t clk_start, clk_stop;
    cudaError_t cudaStatus;
    cudnnStatus_t status;
    cudnnSoftmaxMode_t softmax_mode;
    cudnnSoftmaxAlgorithm_t softmax_algo;
    cudnnHandle_t handle_;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
};


