%%writefile max.cc
#include "connv.h"
#include "cudnn_utility.h"

ConvolutionForward::ConvolutionForward(int batch, int channel, int height, int width, int filter_batch,
                                       int filter_channel, int filter_height, int filter_width, int padding, 
                                       int stride, int dilation, char *mode, char *preference) : batch(batch), channel(channel), height(height),
                                       width(width), filter_batch(filter_batch), filter_channel(filter_channel), 
                                       filter_height(filter_height), filter_width(filter_width), padding(padding),
                                       stride(stride), dilation(dilation), mode(mode), preference(preference) {}
 
void ConvolutionForward::FreeMemory() {
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
    std::cout << " Device input memory deallocation error\n" << std::endl; 
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device output memory deallocation error\n" << std::endl; 
  }

  cudaStatus = cudaFree(DeviceFilterTensor);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device filter memory deallocation error\n" << std::endl;
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
    std::cout << " Unable to Destroy filter Descriptor\n" << std::endl;
  }

  status = cudnnDestroyConvolutionDescriptor(convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy convolution Descriptor\n" << std::endl; 
  }

  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device workspace memory deallocation error\n" << std::endl; 
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy handle\n" << std::endl;
  }
}

int ConvolutionForward::ConvolutionForwardApiCall() {
  // Generating random input_data 
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
  std::cout << "Created cuDNN handle\n" << std::endl;
  
  //! Device Memory allocation for input data
  cudaStatus = cudaMalloc(&DeviceInputTensor, input_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device Input Memory allocation error \n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! Device Memory allocation for filter data
  cudaStatus = cudaMalloc(&DeviceFilterTensor, filter_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device Filter Memory allocation error \n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  //! Copying data from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, input_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    std::cout << "\n!!!! Setting up values on device for input tensor failed\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMemcpy(DeviceFilterTensor, HostFilterTensor, filter_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for filter tensor failed\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating input descriptor error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting input descriptor error\n" << std::endl;  
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
    std::cout << " Setting filter Descriptor error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  padding_height = padding;
  padding_width = padding;
  stride_height = stride;
  stride_width = stride;
  dilation_height = dilation;
  dilation_width = dilation;

  /**
   * CUDNN_CONVOLUTION
   *     In this configuration, a convolution operation 
   *     will be done when applying the filter to the images.    
   * CUDNN_CROSS_CORRELATION
   *     In this configuration, a cross-correlation operation 
   *     will be done when applying the filter to the images.
   */
  if (mode == "cross_correlation") {
    convolution_mode = CUDNN_CROSS_CORRELATION;
    std::cout << "Using convolution_mode : CUDNN_CROSS_CORRELATION" << std::endl; 
  }

  else {
    convolution_mode = CUDNN_CONVOLUTION;
    std::cout << "Using convolution_mode : CUDNN_CONVOLUTION" << std::endl;
  }
  
  status = cudnnCreateConvolutionDescriptor(&convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating convolution Descriptor error\n";  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetConvolution2dDescriptor(convolution_desc, padding_height, padding_width, 
                                           stride_height, stride_width, dilation_height, 
                                           dilation_width, convolution_mode, data_type);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting Convolution Descriptor error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnGetConvolution2dForwardOutputDim(convolution_desc, input_desc, filter_desc,
                                                 &output_batch, &output_channel, &output_height, 
                                                 &output_width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " GetConvolution2dForwardOutputDim error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  int output_size = output_batch * output_channel * (output_height) * (output_width);
  int output_size_bytes = output_size * sizeof(float);
  
  HostOutputTensor = new float[output_size];

  cudaStatus = cudaMalloc(&DeviceOutputTensor, output_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device output Memory allocation error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating Output descriptor error\n" << std::endl;
    return EXIT_FAILURE;
  }
  
  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type, output_batch, 
                                      output_channel, output_height, output_width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting Output descriptor error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  } 

  alpha = ALPHA_INITIAL_VALUE;
  beta = BETA_INITIAL_VALUE;

  /**
   * CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
   *     In this configuration, the routine
   *     cudnnGetConvolutionForwardAlgorithm() is guaranteed to return an
   *     algorithm that does not require any extra workspace to be provided by the user.
   * CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
   *     In this configuration, the routine
   *     cudnnGetConvolutionForwardAlgorithm() will return the fastest
   *     algorithm regardless how much workspace is needed to execute it.
   * CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
   *     In this configuration, the routine
   *     cudnnGetConvolutionForwardAlgorithm() will return the fastest
   *     algorithm that fits within the memory limit that the user provided.
   */
  if (preference == "no_workspace") {
    data_preference =  CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    std::cout << "Using data_preference : CUDNN_CONVOLUTION_FWD_NO_WORKSPACE" << std::endl;
  }

  else if (preference == "specify_workspace_limit") {
    data_preference =  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
    std::cout << "Using data_preference : CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT" << std::endl;
  }

  else {
    data_preference =  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    std::cout << "Using data_preference : CUDNN_CONVOLUTION_FWD_PREFER_FASTEST" << std::endl;
  }
  
  // algorithm
  status = cudnnGetConvolutionForwardAlgorithm(handle_, input_desc, filter_desc, convolution_desc, 
                                               output_desc, data_preference,
                                               /*GPUmemoryLimitInbytes*/ 0, &convolution_algo);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Getting Convolution Forward Algorithm error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  // workspace
  size_t workspace_size;
  status = cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc, filter_desc, convolution_desc, 
                                                   output_desc, convolution_algo, &workspace_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Getting Convolution Forward Workspace size error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMalloc(&workspace_data, workspace_size);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device workspace_size memory allocation failed\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }
      
  std::cout << "\nWorkspace size: " << workspace_size << std::endl;
  std::cout << std::endl;
  
  //! the convolution
  clk_start=clock();
      
  status = cudnnConvolutionForward(handle_, &alpha, input_desc, DeviceInputTensor, filter_desc, 
                                   DeviceFilterTensor, convolution_desc, convolution_algo, 
                                   workspace_data, workspace_size, &beta, output_desc, DeviceOutputTensor);
  
  clk_stop=clock();
      
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " API failed to execute\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  std::cout << "\nInput n*c*h*w: " << input_size << 
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, input_size) << std::endl;
  
  std::cout << "\nOutput_data:" << std::endl;

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, output_size_bytes, cudaMemcpyDeviceToHost);
  if( cudaStatus != cudaSuccess) {
    std::cout <<" Copying data from device to host failed\n" << std::endl;
    return EXIT_FAILURE;
  }

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
  char *mode, *preference;
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

    else if (!(cmd_argument.compare("-mode")))
      mode = (argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-preference")))
      preference = (argv[loop_count + 1]);
  }

  ConvolutionForward convolutionforward(batch, channel, height, width, filter_batch, filter_channel, 
                          filter_height, filter_width, padding, stride, dilation, mode, preference);
  convolutionforward.ConvolutionForwardApiCall();
}
