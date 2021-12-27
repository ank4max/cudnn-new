#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>

int main(int argc, char** argv) {    
  // Reading values for input parameters using command line arguments 
  for (int i = 0;i <= 5; i++) {
    std::cout << argv[i] << std::endl;
  }
  int n, c, h, w;
  std::string mode_set;
  for (int i = 1; i <= 5; i++) {
    int len = sizeof(argv[i]);
    if (argv[i][1] == 'n') {
      n = atoi(argv[i] + 2);
    }
    else if (argv[i][1] == 'c') {
      c = atoi(argv[i] + 2);
    }
    else if (argv[i][1] == 'h') {
      h = atoi(argv[i] + 2);
    }
    else if (argv[i][1] == 'w') {
      w = atoi(argv[i] + 2);
    }
    else if (argv[i][1] == 'a') {
      mode_set = argv[i] + 2; 
    }
  }

  // Generating random input_data 
  int size = n*c*h*w;
  int InputData[size];
  for (int i = 0; i < size; i++) {
    InputData[i] = rand() % 10;
  }
  int num_GPUs;
  cudaError_t cudaStatus;
  cudnnStatus_t status;
  
  cudaStatus = cudaGetDeviceCount(&num_GPUs);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU count error");
    return EXIT_FAILURE;   
  }
  
  std::cout << "Found " << num_GPUs << " GPUs." << std::endl;
  
  //setting device
  cudaStatus = cudaSetDevice(0);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU set device  error");
    return EXIT_FAILURE;   
  }
  
  int device; 
  struct cudaDeviceProp devProp;
  cudaStatus = cudaGetDevice(&device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device read  error");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaGetDeviceProperties(&devProp, device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device properties  error\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

  cudnnHandle_t handle_;
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Created cuDNN handle" << std::endl;

  // Create the tensor descriptor
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  cudnnTensorDescriptor_t x_desc;
  status = cudnnCreateTensorDescriptor(&x_desc);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error\n");
    return EXIT_FAILURE;   
  }
 
  // create the tensor
  float *InputArray;
  cudaStatus = cudaMallocManaged(&InputArray, size * sizeof(float));
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
                    
  for (int i = 0; i < size; i++) {
    InputArray[i] = InputData[i] * 1.00f;
  }

  std::cout << "Original array: "; 
  for (int i = 0; i < size; i++) {
    std::cout << InputArray[i] << " ";
  }
  std::cout << std::endl;

  // create activation function descriptor
  float alpha[c] = {1};
  float beta[c] = {0.0};
  cudnnActivationDescriptor_t activation;
  cudnnActivationMode_t mode;
 
  // Initializing activation mode 
  if (mode_set == "tanh") {
    mode = CUDNN_ACTIVATION_TANH;
  }
  else if (mode_set == "sigmoid") {
    mode = CUDNN_ACTIVATION_SIGMOID;
  }
  else if (mode_set == "relu") {
    mode = CUDNN_ACTIVATION_RELU;
  }
 
  cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
  status = cudnnCreateActivationDescriptor(&activation);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetActivationDescriptor(activation, mode, prop, 0.0f);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("Setting activation  descriptor error\n");
    return EXIT_FAILURE;   
  }

  clock_t start, stop;
  start=clock();

  status = cudnnActivationForward(
        handle_,
        activation,
        alpha,
        x_desc,
        InputArray,
        beta,
        x_desc,
        InputArray 
    ); 

  stop=clock();
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" kernel execution error\n");
    return EXIT_FAILURE;   
  }
  double flopsCoef = 2.0;

  std::cout << "\nInput n*c*h*w: " << size <<"\nLatency: " <<  ((double)(stop-start)) / CLOCKS_PER_SEC <<
  "\nThroughput: " << (1e-9 * flopsCoef * size)/(stop - start) << "\n";

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to uninitialize handle");
    return EXIT_FAILURE;   
  }
  std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;

  std::cout << "New array: ";
  for(int i=0;i<size;i++) {
    std::cout << InputArray[i] << " ";
  }
  std::cout << std::endl;

  cudaStatus = cudaFree(InputArray);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
    return 0;
}

