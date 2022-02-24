%%writefile max1.cc
#include "batchnorm.h"
#include "cudnn_utility.h"

#define ExponentialAverageFactor 0.5
#define EPSILON 0.001

BatchNormalizationForward::BatchNormalizationForward(int batch, int channel, int height, int width, char *mode)
                                                     : batch(batch), channel(channel), height(height), width(width),
                                                     mode(mode) {}

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
    std::cout << "Device input memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device output memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_scale);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device scale memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_offset);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device offset memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_running_mean);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device running_mean memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_running_var);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device running_var memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_saved_mean);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device saved_mean memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_saved_inv_var);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device saved_inv_var memory deallocation error\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(mean_descriptor);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy mean Descriptor\n" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Unable to uninitialize handle\n" << std::endl;
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
    std::cout << " Unable to initialize handle\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "Created cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating output tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting output tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    std::cout << " Memory allocation on device for input tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    std::cout << " Memory allocation on device for output tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying Input values from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for Input tensor failed\n" << std::endl;
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
  if (mode == "per_activation") {
    bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    std::cout << "\nUsing batchnorm mode : CUDNN_BATCHNORM_PER_ACTIVATION\n";
  }

  else if (mode == "spatial") {
    bn_mode = CUDNN_BATCHNORM_SPATIAL;
    std::cout << "\nUsing batchnorm mode : CUDNN_BATCHNORM_SPATIAL\n";
  }
  
  else {
    bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    std::cout <<"\nUsing batchnorm mode : CUDNN_BATCHNORM_SPATIAL_PERSISTENT\n";
  }

  alpha = ALPHA_INITIAL_VALUE;
  beta= BETA_INITIAL_VALUE;

  status = cudnnCreateTensorDescriptor(&mean_descriptor);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating mean descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(mean_descriptor, data_format, data_type,
                                      1, mean_size, 1, 1);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting mean descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  scale = new float[mean_size];
  offset = new float[mean_size];
  running_mean = new float[mean_size];
  running_var = new float[mean_size];

  //! initialize scale, offset, running_mean, running_var
  for (int index = 0; index < mean_size; index++) {
    scale[index] = INITIAL_VALUE;
    offset[index] = INITIAL_VALUE;
    running_mean[index] = INITIAL_VALUE;
    running_var[index] = INITIAL_VALUE;
  }

  cudaStatus = cudaMalloc(&device_scale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for scale\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_offset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for offset\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for running_mean\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for running_var\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_scale, scale, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_offset, offset, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_mean, running_mean, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_var, running_var, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_inv_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  clk_start=clock();
  status = cudnnBatchNormalizationForwardTraining(handle_, bn_mode, &alpha, &beta, input_desc, DeviceInputTensor, output_desc,
                                                  DeviceOutputTensor, mean_descriptor, device_scale,device_offset, 
                                                  ExponentialAverageFactor, device_running_mean, device_running_var, 
                                                  EPSILON, device_saved_mean, device_saved_inv_var);

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Kernel execution error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying data from device to host
  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes,
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on host for output tensor failed\n" << std::endl;
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
  char *mode;

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
      mode = (argv[loop_count + 1]);
  }

  BatchNormalizationForward batchnormalization(batch, channel, height, width, mode);
  batchnormalization.BatchNormalizationForwardApiCall();
}

