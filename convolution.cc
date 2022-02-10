#include "convolution.h"

//print function

Convolution::Convolution(int batch, int channel, int height, int width)
    : batch(batch), channel(channel), height(height), width(width) {}

Convolution::FreeMemory() {
  
  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(output_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyConvolutionDescriptor(convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy convolution Descriptor\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(kernel_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyFilterDescriptor(kernel_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy Filter Descriptor\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(input_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroy(handle);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy handle\n");
    return EXIT_FAILURE;   
  }
  
}

Convolution::ConvolutionForwardApiCall() {
  // Generating random input_data 
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);
  
  float InputData[size];
  for (int i = 0; i < size; i++) {
    InputData[i] = rand() % 255;
  }

  // Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Created cuDNN handle" << std::endl;
  
  // create the tensor descriptor
  std::cout << "Creating  Input descriptor" << std::endl;
  
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
  
  //! Device Memory allocation for input data
  float *in_data;
  cudaStatus = cudaMallocManaged(&in_data, size_bytes);
  
  if( cudaStatus != cudaSuccess) {
    printf(" Device Memory allocation error \n");
    return EXIT_FAILURE;   
  }
  
  //! Copying Input data from host to device
  cudaStatus = cudaMemcpy(in_data, InputData, size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf(" failed to copy input data to device\n");
    return EXIT_FAILURE;   
  }
  
  //!  Filter Descriptor
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;
  
  status  = cudnnCreateFilterDescriptor(&filter_desc); 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating filter Descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetFilter4dDescriptor(filter_desc, format, dtype, filt_k, filt_c, filt_h, filt_w);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set filter Descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  //! Device memory allocation for filter data
  float *filt_data;
  cudaStatus = cudaMalloc(&filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float));
  
  if( cudaStatus != cudaSuccess) {
    printf(" Device memory allocation error \n");
    return EXIT_FAILURE;   
  }
  
  
  //! convolution Descriptor
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;
  
  status = cudnnCreateConvolutionDescriptor(&convolution_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating convolution Descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetConvolution2dDescriptor(convolution_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w,
                                           CUDNN_CROSS_CORRELATION, dtype);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting Convolution Descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  // output Descriptor
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  status = cudnnGetConvolution2dForwardOutputDim(convolution_desc, input_desc, filter_desc,
                                                 &out_n, &out_c, &out_h, &out_w);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting GetConvolution2dForwardOutputDim error\n");
    return EXIT_FAILURE;   
  }
  
  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;
  
  status = cudnnCreateTensorDescriptor(&output_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating Output Tensor descriptor error\n");
    return EXIT_FAILURE;
  
  status = cudnnSetTensor4dDescriptor(output_desc, format, dtype, out_n, out_c, out_h, out_w);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting Output Tensor descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  float *out_data;
  cudaStatus = cudaMallocManaged(&out_data, out_n * out_c * out_h * out_w * sizeof(float));  

  if( cudaStatus != cudaSuccess) {
    printf(" device Memory allocation failed\n");
    return EXIT_FAILURE;   
  }
    
  // algorithm
  status = cudnnGetConvolutionForwardAlgorithm(handle_,input_desc, filter_desc, convolution_desc, output_desc,
                                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Forward Algorithm  error\n");
    return EXIT_FAILURE;   
  }

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;
  
  // workspace
  size_t ws_size;
  status = cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc, filter_desc, convolution_desc, output_desc,
                                                   algo, &ws_size);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Forward Workspace size error\n");
    return EXIT_FAILURE;   
  }

  float *ws_data;
  cudaStatus = cudaMallocManaged(&ws_data, ws_size);
  if( cudaStatus != cudaSuccess) {
    printf(" device allocation failed for ws_size\n");
    return EXIT_FAILURE;   
  } 
      
  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;
  
  //! the convolution
  clk_start=clock();
      
  status = cudnnConvolutionForward(handle, &alpha, input_desc, in_data, filt_desc, filt_data,
                                   conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data);
  
  clk_stop=clock();
      
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" API faied to execute\n");
    return EXIT_FAILURE;   
  }
  
  double flopsCoef = 2.0;
  std::cout << "Input n*c*h*w: " << size << 
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;
  
  cudaStatus = cudaDeviceSynchronize();
  if( cudaStatus != cudaSuccess) {
    printf(" Device synchronization error\n");
    return EXIT_FAILURE;   
  }
  
  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, batch, channel, height, width);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);
  
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
  }

  Convolution convolution(batch, channel, height, width);
  convolution.ConvolutionForwardApiCall();
}
