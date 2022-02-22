%%writefile softmaxback.cc
#include "softmaxbackward.h"
#include "cudnn_utility.h"

SoftmaxBackward::SoftmaxBackward(int batch, int channel, int height, int width, char *mode)
    : batch(batch), channel(channel), height(height), width(width), mode(mode) {}

void SoftmaxBackward::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if (HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device input memory deallocation error" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device output memory deallocation error" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Unable to uninitialize handle" << std::endl;
  }
}

int SoftmaxBackward::SoftmaxBackwardApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  HostInputTensor = new float[size];
  HostOutputTensor = new float[size];

  Util::InitializeNormalizedInputTensor(HostInputTensor, size);

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

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
   if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating ouput tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting output tensor descriptor error \n" << std::endl;
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

  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for input tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  alpha = ALPHA_INITIAL_VALUE;
  beta = BETA_INITIAL_VALUE;

  /*
   * CUDNN_SOFTMAX_MODE_INSTANCE
   *    The softmax operation is computed per image (N) across the dimensions C,H,W.
   * CUDNN_SOFTMAX_MODE_CHANNEL
   *    The softmax operation is computed per spatial location (H,W) per image (N) across
   *    the dimension C.
   */
  if (mode == "softmax_mode_instance") {
    softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  }

  else if (mode == "softmax_mode_channel") {
    softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
  } 

  /* CUDNN_SOFTMAX_FAST
   *    This implementation applies the straightforward softmax operation.
   * CUDNN_SOFTMAX_ACCURATE
   *    This implementation scales each point of the softmax input domain by its maximum
   *    value to avoid potential floating point overflows in the softmax evaluation.
   * CUDNN_SOFTMAX_LOG
   *    This entry performs the Log softmax operation, avoiding overflows by scaling each
   *    point in the input domain as in CUDNN_SOFTMAX_ACCURATE
   */
  cudnnSoftmaxAlgorithm_t softmax_algo = CUDNN_SOFTMAX_FAST;

  clk_start=clock();
  status = cudnnSoftmaxBackward(handle_,                     //handle
                               softmax_algo,                //softmax algo
                               softmax_mode,                //softmax mode
                               &alpha,                      //alpha
                               input_desc,                  //xDesc
                               DeviceInputTensor,           //x
                               input_desc, 
                               DeviceInputTensor,
                               &beta,                       //beta
                               output_desc,                 //yDesc
                               DeviceOutputTensor);         //y

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Kernel execution error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on host for output tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "Input n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  cudaStatus = cudaDeviceSynchronize();
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device synchronization error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nOutput_data:" << std::endl;
  Util::PrintTensor(HostOutputTensor, batch, channel, height, width);

  FreeMemory();

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Reading values for input parameters using command line arguments
  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  int batch, channel, height, width;
  char *mode;

  //! reading cmd line arguments and initializing the required parameters
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

    else if (!(cmd_argument.compare("-mode")))
      mode = (argv[loop_count + 1]);
  }

  SoftmaxBackward softmaxbackward(batch, channel, height, width, mode);
  softmaxbackward.SoftmaxBackwardApiCall();
}
