#include "pooling.h"
#include "cudnn_utility.h"

Pooling::Pooling(int batch, int channel, int height, int width, int window, int padding, int stride, char* pooling_mode)
    : batch(batch), channel(channel), height(height), width(width),
      window(window), padding(padding), stride(stride), pooling_mode(pooling_mode) {}

int Pooling::FreeMemory() {
  if (HostInputTensor)
    delete[] HostInputTensor;

  if (HostOutputTensor)
    delete[] HostOutputTensor;

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device memmory deallocation error\n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device memmory deallocation error\n");
    return EXIT_FAILURE;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
    return EXIT_FAILURE;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
    return EXIT_FAILURE;
  }

  status = cudnnDestroyPoolingDescriptor(pooling_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
    return EXIT_FAILURE;
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
    return EXIT_FAILURE;
  }
}

int Pooling::PoolingForwardApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  int output_height = 1 + (height + padding * 2 - window) / stride;
  int output_width = 1 + (width + padding * 2 - window) / stride;

  int output_size = batch * channel * output_height * output_width;
  int output_size_bytes = output_size * sizeof(float);

  std::cout << output_size << std::endl;

  HostInputTensor = new float[size];
  HostOutputTensor = new float[output_size];

  Util::InitializeInputTensor(HostInputTensor, size);

  std::cout << "\nInput_data:" << std::endl;
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);

  // Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    return EXIT_FAILURE;
  }
  std::cout << "Created cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor x error\n");
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, format, dtype, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor x error\n");
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
   if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor  y error\n");
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, format, dtype, batch, channel, output_height, output_width);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error y \n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
     return EXIT_FAILURE;
  }

  vertical_padding = padding;
  horizontal_padding = padding;
  window_height = window;
  window_width = window;
  vertical_stride = stride;
  horizontal_stride = stride;

  if (pooling_mode == "Max")
    mode = CUDNN_POOLING_MAX;

  else if (pooling_mode == "Average_padding_included")
    mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  else if (pooling_mode == "Average_padding_excluded")
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  status = cudnnCreatePoolingDescriptor(&pooling_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    return EXIT_FAILURE;
  }

  status = cudnnSetPooling2dDescriptor(pooling_desc,             //descriptor handle
                                       mode,                     //pooling mode
                                       CUDNN_PROPAGATE_NAN,      //NAN propagation mode
                                       window_height,            //window height
                                       window_width,             //window width
                                       vertical_padding,         //vertical padding
                                       horizontal_padding,       //horizontal padding
                                       vertical_stride,          //vertical stride
                                       horizontal_stride);       //horizontal stride
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting  activation descriptor error\n");
    return EXIT_FAILURE;
  }

  clk_start=clock();
  status = cudnnPoolingForward(handle_,              //handle
                               pooling_desc,         //poolingdescripor
                               &alpha,               //alpha
                               input_desc,           //xDesc
                               DeviceInputTensor,    //x
                               &beta,                //beta
                               output_desc,          //yDesc
                               DeviceOutputTensor);  //y

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Kernel execution error\n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes, cudaMemcpyDeviceToHost);
   if (cudaStatus != cudaSuccess) {
     fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
     return EXIT_FAILURE;
  }

  std::cout << "Input n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  cudaStatus = cudaDeviceSynchronize();
  if( cudaStatus != cudaSuccess) {
    printf(" Device synchronization error\n");
    return EXIT_FAILURE;
  }

  std::cout << "\nOutput_data:" << std::endl;
  Util::PrintTensor(HostOutputTensor, batch, channel, output_height, output_width);
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

  int batch, channel, height, width, window, padding, stride;
  char* pooling_mode;

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

    else if (!(cmd_argument.compare("-window")))
      window = std::atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-padding")))
      padding = std::atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-stride")))
      stride = std::atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-pooling_mode")))
      pooling_mode = argv[loop_count + 1];

  }

  Pooling pooling(batch, channel, height, width, window, padding, stride, pooling_mode);
  pooling.PoolingForwardApiCall();
}