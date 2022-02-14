#include "pooling.h"

void PrintArray(float *Array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << Array[i] << " ";
  }
  std::cout << std::endl;
}

Pooling::Pooling(int batch, int channel, int height, int width)
    : batch(batch), channel(channel), height(height), width(width) {}

Pooling::FreeMemory() {
  cudaStatus = cudaFree(input);
  if( cudaStatus != cudaSuccess) {
    printf("Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(output);
  if( cudaStatus != cudaSuccess) {
    printf("Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
    return EXIT_FAILURE;   
  }
}

Pooling::PoolingForwardApiCall() {
  // Generating random input_data 
  int size = batch * channel * height * width;

  // Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Created cuDNN handle" << std::endl;  

  int size_bytes = size * sizeof(float);

  // create the tensor descriptor
  std::cout << "Creating descripter" << std::endl;
  

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
  
  status = cudnnSetTensor4dDescriptor(output_desc, format, dtype, batch, channel, height, width);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error y \n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&input, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&output, size_bytes);
  if(cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }

  // initializing data    
  for (int i = 0; i < size; i++) {
    input[i] = (rand() % 255) * 1.0;
  }
  PrintArray(input, size, "Original array: ");

  cudnnPoolingDescriptor_t pooling_desc;
  status = cudnnCreatePoolingDescriptor(&pooling_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetPooling2dDescriptor(pooling_desc,         //descriptor handle
                                       CUDNN_POOLING_MAX,    //mode - max pooling
                                       CUDNN_PROPAGATE_NAN,  //NaN propagation mode
                                       3,    //window height
                                       3,    //window width
                                       0,    //vertical padding
                                       0,    //horizontal padding
                                       1,    //vertical stride
                                       1);   //horizontal stride
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting  activation descriptor error\n");
    return EXIT_FAILURE;   
  }

  
  clk_start=clock();
  status = cudnnPoolingForward(handle_,         //handle
                               pooling_desc,    //poolingdescripor
                               &alpha,          //alpha
                               input_desc,      //xDesc
                               input,           //x
                               &beta,           //beta
                               output_desc,     //yDesc
                               output);         //y 
    
  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Kernel execution error\n");
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

  PrintArray(output, size, "output: ");
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

  Pooling pooling(batch, channel, height, width);
  pooling.PoolingForwardApiCall();
}