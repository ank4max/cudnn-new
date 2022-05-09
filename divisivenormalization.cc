#include "cudnn_DivisiveNormalizationForward_test.h"
#include "cudnn_utility.h"

#define ExponentialAverageFactor 0.5
#define EPSILON 0.001

DivisiveNormalizationForward::DivisiveNormalizationForward(int batch,
    int channel, int height, int width) : batch(batch),
    channel(channel), height(height), width(width) {

        divnorm_mode = CUDNN_DIVNORM_PRECOMPUTED_MEANS;
    }

void DivisiveNormalizationForward::FreeMemory() {
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
    std::cout << "Device input memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device output memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(means);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device means memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(temp);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device temp memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(temp2);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device temp2 memory deallocation error\n" << std::endl;
  }


  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor\n" << std::endl;
  }

  status = cudnnDestroyLRNDescriptor(DivisiveNorm_descriptor);
   if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy DivisiveNorm  Descriptor\n" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Unable to uninitialize handle\n" << std::endl;
  }
}

int DivisiveNormalizationForward::DivisiveNormalizationForwardApiCall() {
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

  

  alpha = ALPHA_INITIAL_VALUE;
  beta= BETA_INITIAL_VALUE;

  status = cudnnCreateLRNDescriptor(&DivisiveNorm_descriptor);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating Lrn descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  unsigned  lrnN;
  double  lrnAlpha;
  double  lrnBeta;
  double  lrnK;
  status = cudnnGetLRNDescriptor(DivisiveNorm_descriptor, &lrnN, &lrnAlpha, &lrnBeta, &lrnK);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Getting LRN descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }


  status = cudnnSetLRNDescriptor(DivisiveNorm_descriptor, lrnN, lrnAlpha, lrnBeta, lrnK);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting LRN descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&means, size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for means failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }


  cudaStatus = cudaMalloc(&temp, size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for temp failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&temp2, size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for temp2 failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Setting mode for DivisiveNormalization
  divisivenorm_mode = CUDNN_DIVNORM_PRECOMPUTED_MEANS;


  /**
   * This function performs the forward spatial DivisiveNormalization layer computation. \n
   * Note that DivisiveNormalization only implements the x/max(c, sigma_x) portion of the computation, 
   * where sigma_x is the variance over the spatial neighborhood of x. 
   */

  /**
   * The API returns the following status :
   * CUDNN_STATUS_SUCCESS - The computation was performed successfully. \n
   * CUDNN_STATUS_BAD_PARAM -  At least one of the following conditions are met: 
   *  One of the tensor pointers x, y, temp, temp2 is NULL.
   *  Number of input tensor or output tensor dimensions is outside of [4,5] range. 
   *  A mismatch in dimensions between any two of the input or output tensors.
   *  For in-place computation when pointers x == y, a mismatch in strides between the input data and output data tensors.
   *  Alpha or beta pointer is NULL.
   *  LRN descriptor parameters are outside of their valid ranges.
   *  Any of the tensor strides are negative.
   * CUDNN_STATUS_UNSUPPORTED - The function does not support the provided configuration, 
   * for example, any of the input and output tensor strides mismatch (for the same dimension) is a non-supported configuration.
   */


  //! API call
  clk_start=clock();

  status = cudnnDivisiveNormalizationForward(handle_, DivisiveNorm_descriptor,
                divisivenorm_mode, &alpha, input_desc, DeviceInputTensor, 
                means, temp, temp2, &beta, output_desc,  
                DeviceOutputTensor);

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

  int batch, channel, height, width, status;
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

  }

  DivisiveNormalizationForward divisivenormalizationforward(batch, channel, height, width);
  status = divisivenormalizationforward.DivisiveNormalizationForwardApiCall();
  return status;
}
