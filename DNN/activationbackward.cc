%%writefile max.cc
#include "activation.h"
#include "cudnn_utility.h"

ActivationBackward::ActivationBackward(int batch, int channel, int height, int width, char* activation_mode)
    : batch(batch), channel(channel), height(height), width(width),
      activation_mode(activation_mode) {}

void ActivationBackward::FreeMemory() {
  if(HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if(HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device input memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device output memory deallocation error\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor\n" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Unable to uninitialize handle\n" << std::endl;
  }
}

int ActivationBackward::ActivationBackwardApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  //! Initializing input data
  HostInputTensor = new float[size];
  HostOutputTensor= new float[size];

  Util::InitializeActivationTensor(HostInputTensor, size);

  //! Printing initial array before activation
  std::cout << "\nInput array: ";
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);
  std::cout << std::endl;

  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to initialize handle\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "\nCreated cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, format, dtype, batch, channel, height, width);
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

  status = cudnnSetTensor4dDescriptor(output_desc, format, dtype, batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting output tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Device memory allocation for Input and Output Arrays
  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for input\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for output\n" << std::endl;
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

  float alpha[channel] = {ALPHA_INITIAL_VALUE};
  float beta[channel] = {BETA_INITIAL_VALUE};

  //! Initializing activation mode
  if (activation_mode == "tanh") {
    mode = CUDNN_ACTIVATION_TANH;
  }
  else if (activation_mode == "sigmoid") {
    mode = CUDNN_ACTIVATION_SIGMOID;
  }
  else if (activation_mode == "relu") {
    mode = CUDNN_ACTIVATION_RELU;
  }

  propagation = CUDNN_NOT_PROPAGATE_NAN;

  //! Setting activation descriptor
  status = cudnnCreateActivationDescriptor(&activation_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating activation descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetActivationDescriptor(activation_desc, mode, propagation, RELU_CLIPPING_THREASHOLD);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Setting activation  descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! API call
  clk_start=clock();

  status = cudnnActivationBackward(handle_, activation_desc, alpha, input_desc, DeviceInputTensor,
                                  input_desc, DeviceInputTensor, output_desc, DeviceOutputTensor,
                                  beta, output_desc, DeviceOutputTensor);

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " kernel execution error\n" << std::endl;
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

  std::cout << "\nOutput array: ";
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
  char* activation_mode;

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

    else if (!(cmd_argument.compare("-activation_mode")))
      activation_mode = argv[loop_count + 1];
  }

  ActivationBackward activationbackward(batch, channel, height, width, activation_mode);
  activationbackward.ActivationBackwardApiCall();
}
