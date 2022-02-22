%%writefile max.cc
#include "batchnorm.h"
#include "cudnn_utility.h"

#define ONE 1.0
#define ZERO 0.0
#define ExponentialAverageFactor 0.5
#define EPSILON 0.001

BatchNormalizationForward::BatchNormalizationForward(int batch, int channel, int height, int width, char* batchnorm_mode)
    : batch(batch), channel(channel), height(height), width(width), batchnorm_mode(batchnorm_mode) {}

void BatchNormalizationForward::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if (HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  if (scale) {
    delete[] scale;
    scale = nullptr;
  }

  if (offset) {
    delete[] offset;
    offset = nullptr;
  }

  if (running_mean) {
    delete[] running_mean;
    running_mean = nullptr;
  }

  if (running_var) {
    delete[] running_var;
    running_var = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device output memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_scale);
  if( cudaStatus != cudaSuccess) {
    printf(" Device scale memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_offset);
  if( cudaStatus != cudaSuccess) {
    printf(" Device offset memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_running_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device running_mean memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_running_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device running_var memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_saved_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device saved_mean memory deallocation error\n");
  }

  cudaStatus = cudaFree(device_saved_inv_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device saved_inv_var memory deallocation error\n");
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(mean_descriptor);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy mean Descriptor\n");
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
}

int BatchNormalizationForward::BatchNormalizationForwardApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  int mean_size = channel;
  int mean_size_bytes = mean_size * sizeof(float);

  HostInputTensor = new float[size];
  HostOutputTensor = new float[size];

  Util::InitializeInputTensor(HostInputTensor, size);

  std::cout << "\nInput_data:" << std::endl;
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);

  // Create cudnn handle
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "Created cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating input tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting input tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating output tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting output tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" Memory allocation on device for input tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" Memory allocation on device for output tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying Input values from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for Input tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * CUDNN_BATCHNORM_PER_ACTIVATION
   *    Normalization is performed per-activation. This mode is intended to be used
   *    after non-convolutional network layers. In this mode bnBias and bnScale tensor
   *    dimensions are 1xCxHxW.
   * CUDNN_BATCHNORM_SPATIAL
   *    Normalization is performed over N+spatial dimensions. This mode is intended for
   *    use after convolutional layers (where spatial invariance is desired). In this mode
   *    bnBias, bnScale tensor dimensions are 1xCx1x1.
   * CUDNN_BATCHNORM_SPATIAL_PERSISTENT
   *    This mode is similar to CUDNN_BATCHNORM_SPATIAL but it
   *    can be faster for some tasks.
   */
  
  if (batchnorm_mode == "batchnorm_per_activation") {
      bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  }

  else if (batchnorm_mode == "batchnorm_spatial") {
      bn_mode = CUDNN_BATCHNORM_SPATIAL;
  }
  
  else if (batchnorm_mode =="batchnorm_spatial_persistent") {
     bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }

  float alpha[channel] = {ONE};
  float beta[channel] = {ZERO};

  status = cudnnCreateTensorDescriptor(&mean_descriptor);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating mean descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(mean_descriptor, data_format, data_type,
                                      ONE, mean_size, ONE, ONE);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting mean descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  scale = new float[mean_size];
  offset = new float[mean_size];
  running_mean = new float[mean_size];
  running_var = new float[mean_size];

  //! initialize scale, offset, running_mean, running_var
  for (int index = 0; index < mean_size; index++) {
    scale[index] = ONE;
    offset[index] = ONE;
    running_mean[index] = ONE;
    running_var[index] = ONE;
  }

  cudaStatus = cudaMalloc(&device_scale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed for scale\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_offset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed for offset\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed for running_mean\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed for running_var\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_scale, scale, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for scale tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_offset, offset, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for scale tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_mean, running_mean, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for scale tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_var, running_var, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for scale tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_inv_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  clk_start=clock();
  status = cudnnBatchNormalizationForwardTraining(
                /*handle=*/handle_,
                /*mode=*/bn_mode,
                /**alpha=*/&alpha,
                /**beta=*/&beta,
                /*xDesc=*/input_desc,
                /**x=*/DeviceInputTensor,
                /*yDesc=*/output_desc,
                /**y=*/DeviceOutputTensor,
                /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
                /*bnScaleData=*/device_scale,
                /*bnBiasData=*/device_offset,
                /*exponentialAverageFactor=*/ExponentialAverageFactor,
                /*resultRunningMeanData=*/device_running_mean,
                /*resultRunningVarianceData=*/device_running_var,
                /*epsilon=*/EPSILON,
                /*resultSaveMean=*/device_saved_mean,
                /*resultSaveInvVariance=*/device_saved_inv_var);

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Kernel execution error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying data from device to host
  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes,
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on host for output tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nInput n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  //! Printing the output
  std::cout << "\nOutput_data:" << std::endl;
  Util::PrintTensor(HostOutputTensor, batch, channel, height, width);

  FreeMemory();

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  //! Reading values for input parameters using command line arguments
  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  int batch, channel, height, width;
  char *batchnorm_mode;

  //! Reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-batch")))
      batch = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-channel")))
      channel = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-height")))
      height = std::atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-width")))
      width = std::atoi(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-batchnorm_mode")))
      batchnorm_mode = (argv[loop_count + 1]);
  }

  BatchNormalizationForward batchnormalizationforward(batch, channel, height, width, batchnorm_mode);
  batchnormalizationforward.BatchNormalizationForwardApiCall();
}
