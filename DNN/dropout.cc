%%writefile max41.cc

#include "dropout.h"
#include "cudnn_utility.h"

Dropout::Dropout(int batch, int channel, int height, int width, float drop_rate)
    : batch(batch), channel(channel), height(height), width(width), drop_rate(drop_rate) {}

int Dropout::FreeMemory() {
  
  if(HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }
  
  cudaStatus = cudaFree(states);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree(dropout_reserve_space);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree(DeviceInputTensor);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
	
  status = cudnnDestroyDropoutDescriptor(dropout_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy dropout Descriptor\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnDestroyTensorDescriptor(dropout_in_out_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy drop_in_out_descriptor Descriptor\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
  
}

int Dropout::DropoutForwardApiCall() {
  // Generating random input_data 
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);  

  float HostInputTensor[size];
  Util::InitializeInputTensor(HostInputTensor, size);
  
  //! Create cudnn context
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
	
  std::cout << "\nCreated cuDNN handle" << std::endl;  
  
  status = cudnnCreateDropoutDescriptor(&dropout_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating dropout descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnCreateTensorDescriptor(&dropout_in_out_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating dropout_in_out descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }  
  
  status = cudnnSetTensor4dDescriptor(dropout_in_out_descriptor, format, dtype, batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting dropout_in_out descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
 
  //! Getting states size
  status = cudnnDropoutGetStatesSize(handle_, &dropout_state_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout state size error\n"); 
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  //! Getting reverse space size
  status = cudnnDropoutGetReserveSpaceSize(dropout_in_out_descriptor, &dropout_reserve_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout Reverse size error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  //! Allocate memory for states and reserve space
  cudaStatus = cudaMalloc(&states, dropout_state_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMalloc(&dropout_reserve_space, dropout_reserve_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnSetDropoutDescriptor(dropout_descriptor, handle_, drop_rate, states, dropout_state_size, /*Seed*/time(NULL));
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting dropout descriptor failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  //! Copying data from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes, cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf("\nFailed to copy data from host to device \n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  std::cout << "\nInput \n";
  Util::DropoutPrint(HostInputTensor, height, width, batch * channel);
  
  //! API call
  clk_start=clock();
      
  status = cudnnDropoutForward(handle_, dropout_descriptor, dropout_in_out_descriptor, DeviceInputTensor, dropout_in_out_descriptor,
			       DeviceOutputTensor, dropout_reserve_space, dropout_reserve_size);
  
  clk_stop=clock();
      
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" API faied to execute\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
	
  double flopsCoef = 2.0;
  std::cout << "\nInput n*c*h*w: " << size << 
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;
  
  //! Copying values from device to host
  HostOutputTensor = new float[size];
  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes, cudaMemcpyDeviceToHost);
  if( cudaStatus != cudaSuccess) {
    printf("\nFailed to copy data from Device to host \n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  std::cout << "\nDropout \n";
  Util::DropoutPrint(HostOutputTensor, height, width, batch * channel);
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
  float drop_rate;

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
    
    else if (!(cmd_argument.compare("-drop_rate")))
      drop_rate = std::stof(argv[loop_count + 1]);
  }

  Dropout dropout(batch, channel, height, width, drop_rate);
  dropout.DropoutForwardApiCall();
}
  
  
