#include "dropout.h"
#include "cudnn_utility.h"

DropoutBackward::DropoutBackward(int batch, int channel, int height, int width, float dropout_rate)
    : batch(batch), channel(channel), height(height), width(width),
      dropout_rate(dropout_rate) {}

void DropoutBackward::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if(HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  cudaStatus = cudaFree(states);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  cudaStatus = cudaFree(dropout_reserve_space);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy drop_in_out_descriptor Descriptor\n");
  }

  status = cudnnDestroyDropoutDescriptor(dropout_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy dropout Descriptor\n");
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
}

int DropoutBackward::DropoutBackwardApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  HostInputTensor = new float[size];
  HostOutputTensor = new float[size];

  Util::InitializeInputTensor(HostInputTensor, size);

  std::cout << "\nInput_data:" << std::endl;
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);

  //! Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "\nCreated cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating input descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting input descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * cudnnDropoutGetStatesSize function is used to query the amount of space
   * required to store the states of the random number generators used by
   * cudnnDropoutForward function.
   */
  status = cudnnDropoutGetStatesSize(handle_, &dropout_state_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout state size error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * cudnnDropoutGetReserveSpaceSize function is used to query the amount of
   * reserve needed to run dropout with the input dimensions given by input_desc
   */
  status = cudnnDropoutGetReserveSpaceSize(input_desc, &dropout_reserve_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout Reverse size error\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Allocate memory for states and reserve space
  cudaStatus = cudaMalloc(&states, dropout_state_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error for state\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&dropout_reserve_space, dropout_reserve_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error for dropout_reserve_space\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateDropoutDescriptor(&dropout_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating dropout descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetDropoutDescriptor(dropout_descriptor, handle_, dropout_rate,
                                     states, dropout_state_size, time(NULL));
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting dropout descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error for input_desc\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error for output tensor\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying data from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes,
                          cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf("\nFailed to copy input data from host to device \n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! API call
  clk_start=clock();

  status = cudnnDropoutBackward(handle_, dropout_descriptor, input_desc,
                               DeviceInputTensor, input_desc,
                               DeviceOutputTensor, dropout_reserve_space,
                               dropout_reserve_size);

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" API faied to execute\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nInput n*c*h*w: " << size <<
               "\nLatency: " <<
               ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " <<
               THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes, cudaMemcpyDeviceToHost);
  if( cudaStatus != cudaSuccess) {
    printf("\nFailed to copy output data from Device to host \n");
    FreeMemory();
    return EXIT_FAILURE;
  }

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
  float dropout_rate;

  //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-batch")))
      batch = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-channel")))
      channel = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-height")))
      height = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-width")))
      width = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-dropout_rate")))
      dropout_rate = std::stof(argv[loop_count + 1]);
  }

  DropoutBackward dropoutbackward(batch, channel, height, width, dropout_rate);
  dropoutbackward.DropoutBackwardApiCall();
}
