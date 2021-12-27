#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>

void PrintArray(float *Array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << Array[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {    
  // Reading values for input parameters using command line arguments 
  for (int i = 0;i < 5; i++) {
    std::cout << argv[i] << std::endl;
  }    
  int n, c, h, w;
  std::string a;

  for (int i = 1; i < 5; i++) {
    int len = sizeof(argv[i]);
    if (argv[i][1] == 'n')
      n = atoi(argv[i] + 2);
    else if (argv[i][1] == 'c')
      c = atoi(argv[i] + 2);
    else if (argv[i][1] == 'h')
      h = atoi(argv[i] + 2);
    else if (argv[i][1] == 'w')
      w = atoi(argv[i] + 2);
 
  }

  // Generating random input_data 
  int size = n*c*h*w;
  int InputData[size];
  for (int i = 0; i < size; i++) {
    InputData[i] = rand() % 255;
  }
  int num_GPUs;
  cudaError_t cudaStatus;
  cudnnStatus_t status;
  
  cudaStatus = cudaGetDeviceCount(&num_GPUs);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU count error");
    return EXIT_FAILURE;   
  }
  std::cout << "Found " << numGPUs << " GPUs." << std::endl;
  
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

  // setting parameters for batchnormal API
  auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
  float one = 1.0;
  float zero = 0.0;
  int size_bytes = size * sizeof(float);
  int mean_size = c;
  int mean_size_bytes = mean_size * sizeof(float);

  // create the tensor descriptor
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  std::cout << "Creating descripter" << std::endl;
  
  cudnnTensorDescriptor_t x_desc;
  status = cudnnCreateTensorDescriptor(&x_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor x error\n");
    return EXIT_FAILURE;   
  }
   
  status = cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor x error\n");
    return EXIT_FAILURE;   
  }
  
  cudnnTensorDescriptor_t y_desc;
  status = cudnnCreateTensorDescriptor(&y_desc);
   if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor  y error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetTensor4dDescriptor(y_desc, format, dtype, n, c, h, w);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error y \n");
    return EXIT_FAILURE;   
  }

  float *x, *y, *dy, *dx;
  cudaStatus = cudaMallocManaged(&x, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&y, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&dy, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&dx, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }

  // initializing data    
  for (int i = 0; i < size; i++) {
    x[i] = input_data[i];
  }
  std::cout << "Original array: " << std::endl; 
  for(int i =0 ; i < size;i ++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;

  float alpha[c] = {1};
  float beta[c] = {0.0};

  cudnnTensorDescriptor_t mean_descriptor;
  status = cudnnCreateTensorDescriptor(&mean_descriptor);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating mean descriptor error\n");
    return EXIT_FAILURE;   
  }
  status = cudnnSetTensor4dDescriptor(mean_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/c,
                                        /*image_height=*/1,
                                        /*image_width=*/1);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting mean descriptor error\n");
    return EXIT_FAILURE;   
  }
    
  float *scale, *offset, *dscale, *doffset;
  float *running_mean, *running_var;
  float *saved_mean, *saved_inv_var;
  cudaStatus = cudaMallocManaged(&scale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&offset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&dscale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&doffset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&running_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&running_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&saved_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaMallocManaged(&saved_inv_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }

  // initialize scale, offset, running_mean, running_var
  for (int i = 0; i < mean_size; i++) {
    scale[i] = 1.0;
    offset[i] = 1.0;
    running_mean[i] = 1.0;
    running_var[i] = 1.0;
  }

  cudnnActivationDescriptor_t activation_desc;
  status = cudnnCreateActivationDescriptor(&activation_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetActivationDescriptor(activation_desc,
                                          CUDNN_ACTIVATION_IDENTITY,
                                          CUDNN_PROPAGATE_NAN, 0.0);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting  activation descriptor error\n");
    return EXIT_FAILURE;   
  }

  size_t workspace_size_bytes = 0;
  cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
    /*xDesc=*/x_desc, /*zDesc=*/NULL, /*yDesc=*/y_desc,
    /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
    /*activationDesc=*/activation_desc,
    /*sizeInBytes=*/&workspace_size_bytes);
 
  void *workspace = nullptr;
  if (workspace_size_bytes > 0) {
      cudaMalloc(&workspace, workspace_size_bytes);
  }

  clock_t start, stop;
  start=clock();
  
  size_t reserve_space_size_bytes = 0;
  cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
  /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
  /*activationDesc=*/activation_desc, /*xDesc=*/x_desc,
  /*sizeInBytes=*/&reserve_space_size_bytes);
  char *reserve_space;
  cudaMalloc(&reserve_space, reserve_space_size_bytes);

  status = cudnnBatchNormalizationForwardTraining(
  /*handle=*/handle_,
  /*mode=*/mode,
  /**alpha=*/&one,
  /**beta=*/&zero,
  /*xDesc=*/x_desc,
  /**x=*/x,
  /*yDesc=*/y_desc,
  /**y=*/y,
  /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
  /*bnScaleData=*/scale,
  /*bnBiasData=*/offset,
  /*exponentialAverageFactor=*/0.5,
  /*resultRunningMeanData=*/running_mean,
  /*resultRunningVarianceData=*/running_var,
  /*epsilon=*/0.001,
  /*resultSaveMean=*/saved_mean,
  /*resultSaveInvVariance=*/saved_inv_var);
    
  stop=clock();
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Kernel execution error\n");
    return EXIT_FAILURE;   
  }
  double flopsCoef = 2.0;
  std::cout << "Input n*c*h*w: " << size << 
  "\nLatency: " << ((double)(stop - start))/CLOCKS_PER_SEC <<
  "\nThroughput: " << (1e-9 * flopsCoef * size) / (stop - start) << std::endl;
    
  cudaStatus = cudaDeviceSynchronize();
  if( cudaStatus != cudaSuccess) {
    printf(" Device synchronization error\n");
    return EXIT_FAILURE;   
  }

  PrintArray(y, size, "output: ");


  cudaStatus = cudaFree(x);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(y);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(dy);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(dx);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(scale);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(offset);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(dscale);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(doffset);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(running_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(running_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(saved_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(saved_inv_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(workspace);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(reserve_space);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroy(handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to uninitialize handle\n");
    return EXIT_FAILURE;   
  }
    return 0;
}
