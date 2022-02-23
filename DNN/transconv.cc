%%writefile max.cc
#include "transconv.h"
#include "cudnn_utility.h"


ConvolutionTranspose::ConvolutionTranspose(int batch, int channel, int height,
    int width, int filter_batch, int filter_channel, int filter_height,
    int filter_width, int padding, int stride, int dilation, char *bwd_preference)
    : batch(batch), channel(channel), height(height), width(width),
      filter_batch(filter_batch), filter_channel(filter_channel),
      filter_height(filter_height), filter_width(filter_width),
      padding(padding), stride(stride), dilation(dilation), bwd_preference(bwd_preference) {}

void ConvolutionTranspose::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if (HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  if (HostFilterTensor) {
    delete[] HostFilterTensor;
    HostFilterTensor = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device memmory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device memmory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceFilterTensor);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device memmory deallocation error\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor\n" << std::endl;
  }

  status = cudnnDestroyFilterDescriptor(filter_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy Filter Descriptor\n" << std::endl;
  }

  status = cudnnDestroyConvolutionDescriptor(convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy convolution Descriptor\n" << std::endl;
  }

  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device memmory deallocation error\n" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy handle\n" << std::endl;
  }
}

int ConvolutionTranspose::ConvolutionTransposeApiCall() {
  //! Generating random input_data
  int input_size = batch * channel * height * width;
  int input_size_bytes = input_size * sizeof(float);

  int filter_size = filter_batch * filter_channel * filter_height * filter_width;
  int filter_size_bytes = filter_size * sizeof(float);

  HostInputTensor = new float[input_size];
  HostFilterTensor = new float[filter_size];

  Util::InitializeInputTensor(HostInputTensor, input_size);
  Util::InitializeFilterTensor(HostFilterTensor, filter_size);

  std::cout << "\nInput_data:" << std::endl;
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);

  std::cout << "\nFilter_data:" << std::endl;
  Util::PrintTensor(HostFilterTensor, filter_batch, filter_channel,
                    filter_height, filter_width);

  //! Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to initialize handle\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "\nCreated cuDNN handle" << std::endl;

  //! Device Memory allocation for input data
  cudaStatus = cudaMalloc(&DeviceInputTensor, input_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\nDevice Memory allocation error \n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Device Memory allocation for filter data
  cudaStatus = cudaMalloc(&DeviceFilterTensor, filter_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\nDevice Memory allocation error \n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying data from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, input_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\nFailed to copy input data to device\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(DeviceFilterTensor, HostFilterTensor, filter_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    std::cout << " failed to copy input data to device\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << "\nCreating tensor descriptor x error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << "\nSetting tensor descriptor x error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status  = cudnnCreateFilterDescriptor(&filter_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating filter Descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetFilter4dDescriptor(filter_desc, data_type, data_format, filter_batch,
                                      filter_channel, filter_height, filter_width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Set filter Descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  padding_height = padding;
  padding_width = padding;
  stride_height = stride;
  stride_width = stride;
  dilation_height = dilation;
  dilation_width = dilation;

  status = cudnnCreateConvolutionDescriptor(&convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << "\nCreating convolution Descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetConvolution2dDescriptor(convolution_desc, padding_height, padding_width,
                                           stride_height, stride_width, dilation_height,
                                           dilation_width, CUDNN_CROSS_CORRELATION, data_type);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << "\nSetting Convolution Descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  int output_batch = batch;
  int output_channel = filter_channel;
  int output_height = (height - 1) * stride - 2 * padding_height + filter_height;
  int output_width =  (width - 1) * stride - 2 * padding_width + filter_width;

  int output_size = output_batch * output_channel * output_height * output_width;
  int output_size_bytes = output_size * sizeof(float);

  HostOutputTensor = new float[output_size];

  cudaStatus = cudaMalloc(&DeviceOutputTensor, output_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\nDevice Memory allocation error \n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating Output Tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type, output_batch,
                                      output_channel, output_height, output_width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting Output Tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE
   *     In this configuration, the routine
   *     cudnnGetConvolutionBackwardDataAlgorithm() is guaranteed to return an
   *     algorithm that does not require any extra workspace to be provided by the user.
   * CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
   *     In this configuration, the routine
   *     cudnnGetConvolutionBackwardDataAlgorithm() will return the fastest
   *     algorithm regardless how much workspace is needed to execute it.
   * CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
   *     In this configuration, the routine
   *     cudnnGetConvolutionBackwardDataAlgorithm() will return the fastest
   *     algorithm that fits within the memory limit that the user provided.
   */

  if (bwd_preference == "no_workspace") {
    data_preference =  CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
    std::cout << "using data_preference : CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE" << std::endl;
  }

  else if (bwd_preference == "specify_workspace_limit") {
    data_preference =  CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    std::cout << "using data_preference : CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT" << std::endl;
  }

  else {
    data_preference =  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    std::cout << "using data_preference : CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST" << std::endl;
  }



  alpha = ALPHA_INITIAL_VALUE;
  beta = BETA_INITIAL_VALUE;
  zero = ZERO_INITIAL_VALUE;

  //! Getting algorithm to perform transpose convolution
  status = cudnnGetConvolutionBackwardDataAlgorithm(handle_, filter_desc, input_desc, convolution_desc, output_desc,
                                                    data_preference, zero, &convolution_algo);


  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Convolution Backward Algorithm  error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Getting  workspace size
  size_t workspace_size;
  status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, filter_desc, input_desc, convolution_desc, output_desc,
                                                        convolution_algo, &workspace_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Convolution Backward Workspace size error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&workspace_data, workspace_size);
  if( cudaStatus != cudaSuccess) {
    std::cout << " device allocation failed for ws_size\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nWorkspace size: " << workspace_size << std::endl;
  std::cout << std::endl;

  //! APi call to perform tranpose convolution

  clk_start = clock();

  status = cudnnConvolutionBackwardData(handle_, &alpha, filter_desc, DeviceFilterTensor, input_desc, DeviceInputTensor,
                                        convolution_desc, convolution_algo, workspace_data, workspace_size, &beta, output_desc,
                                        DeviceOutputTensor);

  clk_stop = clock();

  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << "API error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nInput n*c*h*w: " << input_size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, input_size) << std::endl;

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, output_size_bytes, cudaMemcpyDeviceToHost);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\nCopying data from device to host failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Printing the output
  std::cout << "\nOutput_data:" << std::endl;
  Util::PrintTensor(HostOutputTensor, output_batch, output_channel, output_height, output_width);

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

  int batch, channel, height, width, filter_batch, filter_channel, filter_height, filter_width;
  int padding, stride, dilation;
  char *bwd_preference;
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

    else if (!(cmd_argument.compare("-filter_batch")))
      filter_batch = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-filter_channel")))
      filter_channel = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-filter_height")))
      filter_height = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-filter_width")))
      filter_width = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-padding")))
      padding = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-stride")))
      stride = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-dilation")))
      dilation = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-preference")))
      bwd_preference = (argv[loop_count + 1]);
  }

  ConvolutionTranspose convolutiontranspose(batch, channel, height, width, filter_batch, filter_channel,
                                            filter_height, filter_width, padding, stride, dilation, bwd_preference);
  convolutiontranspose.ConvolutionTransposeApiCall();
}

  
