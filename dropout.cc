#include "Dropout.h"

//print function

Dropout::Dropout(int batch, int channel, int height, int width, float drop_rate)
    : batch(batch), channel(channel), height(height), width(width), drop_rate(drop_rate)) {}

int Dropout::FreeMemory() {
  
  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  
}

int Dropout::DropoutForwardApiCall() {
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
  std::cout << "\nCreated cuDNN handle" << std::endl;
  
  
  status = cudnnCreateDropoutDescriptor(&dropout_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating dropout descriptor failed\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnCreateTensorDescriptor(&dropout_in_out_descriptor)
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nCreating dropout_in_out descriptor failed\n");
    return EXIT_FAILURE;   
  }
  
  
  status = cudnnSetTensor4dDescriptor(dropout_in_out_descriptor, format, dtype, batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting dropout_in_out descriptor failed\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnDropoutGetStatesSize(handle_, &dropout_state_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout state size error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnDropoutGetReserveSpaceSize(dropout_in_out_descriptor, &dropout_reserve_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nGet dropout Reverse size error\n");
    return EXIT_FAILURE;   
  }
  
  // Allocate memory for states and reserve space
	cudaMallocManaged(&states,dropout_state_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    return EXIT_FAILURE;   
  }
	cudaMallocManaged(&dropout_reserve_space,dropout_reserve_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    return EXIT_FAILURE;   
  }

	status = cudnnSetDropoutDescriptor(dropout_descriptor, handle_, drop_rate, states, dropout_state_size, /*Seed*/time(NULL));
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("\nSetting dropout descriptor failed\n");
    return EXIT_FAILURE;   
  }
  
  cudaMallocManaged(&d_dropout_out, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    return EXIT_FAILURE;   
  }
  
  cudaMallocmanaged(&d_dx_dropout, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    return EXIT_FAILURE;   
  }
  
  float * d_input;
  cudaStatus = cudaMemcpy(d_input,InputData, size_bytes,cudaMemcpyHostToHost);
  
  
  
