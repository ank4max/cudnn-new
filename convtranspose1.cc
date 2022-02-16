%%writefile max.cc
#include "convolution.h"
#include "cudnn_utility.h"

Convolution::Convolution(int batch, int channel, int height, int width, int filter_batch,
               int filter_channel, int filter_height, int filter_width, int padding, 
               int stride, int dilation) : batch(batch), channel(channel), height(height),
               width(width), filter_batch(filter_batch), filter_channel(filter_channel), 
               filter_height(filter_height), filter_width(filter_width), padding(padding),
               stride(stride), dilation(dilation) {}
 
void Convolution::FreeMemory() {
  if (HostInputTensor)
    delete[] HostInputTensor;
  
  if (HostOutputTensor)
    delete[] HostOutputTensor;
    
  if (HostFilterTensor)
    delete[] HostFilterTensor;

  cudaStatus = cudaFree(DeviceInputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n"); 
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n"); 
  }

  cudaStatus = cudaFree(DeviceFilterTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }
  
  status = cudnnDestroyTensorDescriptor(output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n"); 
  }

  status = cudnnDestroyFilterDescriptor(filter_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy Filter Descriptor\n");
  }

  status = cudnnDestroyConvolutionDescriptor(convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy convolution Descriptor\n"); 
  }

  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n"); 
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy handle\n");
  }
}

int Convolution::ConvolutionBackwardApiCall() {
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
    printf(" Unable to initialize handle\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  std::cout << "\nCreated cuDNN handle" << std::endl;
  
  //! Device Memory allocation for input data
  cudaStatus = cudaMalloc(&DeviceInputTensor, input_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! Device Memory allocation for filter data
  cudaStatus = cudaMalloc(&DeviceFilterTensor, filter_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  //! Copying data from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, input_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf("\nFailed to copy input data to device\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMemcpy(DeviceFilterTensor, HostFilterTensor, filter_size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf(" failed to copy input data to device\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating tensor descriptor x error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting tensor descriptor x error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status  = cudnnCreateFilterDescriptor(&filter_desc); 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating filter Descriptor error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetFilter4dDescriptor(filter_desc, data_type, data_format, filter_batch, 
                                      filter_channel, filter_height, filter_width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set filter Descriptor error\n");  
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
    printf("\nCreating convolution Descriptor error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetConvolution2dDescriptor(convolution_desc, padding_height, padding_width, 
                                           stride_height, stride_width, dilation_height, 
                                           dilation_width, CUDNN_CROSS_CORRELATION, data_type);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting Convolution Descriptor error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  

  int value = (height -1) * stride -2 * padding + (filter_height - 1) + 1;
  int output_size = value * value;
  int output_size_bytes = output_size * sizeof(float);
  
  HostOutputTensor = new float[output_size];

  cudaStatus = cudaMalloc(&DeviceOutputTensor, output_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating Output Tensor descriptor error\n");
    return EXIT_FAILURE;
  }
  
  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type, batch, 
                                      channel, value, value);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting Output Tensor descriptor error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  // algorithm
  //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST

  status = cudnnGetConvolutionBackwardDataAlgorithm(handle_, filter_desc, input_desc, convolution_desc, output_desc,
                                                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0 , &convolution_algo); 
 
 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Backward Algorithm  error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  } 




// workspace
  size_t workspace_size;
  status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, filter_desc, input_desc, convolution_desc, output_desc, 
                                                        convolution_algo, &workspace_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Backward Workspace size error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMalloc(&workspace_data, workspace_size);
  if( cudaStatus != cudaSuccess) {
    printf(" device allocation failed for ws_size\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  } 
      
  std::cout << "\nWorkspace size: " << workspace_size << std::endl;
  std::cout << std::endl;


  //APi call


  clk_start = clock();


  status = cudnnConvolutionBackwardData(handle_, &alpha, filter_desc, DeviceFilterTensor, input_desc, DeviceInputTensor, 
                                        convolution_desc, convolution_algo, workspace_data, workspace_size, &beta, output_desc,
                                        DeviceOutputTensor);

  clk_stop = clock();

  if(status != CUDNN_STATUS_SUCCESS) {
    printf("API error\n");  
    FreeMemory();
    return EXIT_FAILURE;   
  }


  std::cout << "\nInput n*c*h*w: " << input_size << 
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, input_size) << std::endl;

 
  std::cout << "\nOutput_data:" << std::endl;

  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, output_size_bytes, cudaMemcpyDeviceToHost);
  if( cudaStatus != cudaSuccess) {
    printf("\nCopying data from device to host failed\n");
    return EXIT_FAILURE;
  }

  Util::PrintTensor(HostOutputTensor, batch, channel, value, value);
  
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
  }

  Convolution convolution(batch, channel, height, width, filter_batch, filter_channel, 
                          filter_height, filter_width, padding, stride, dilation);
  convolution.ConvolutionBackwardApiCall();
}
