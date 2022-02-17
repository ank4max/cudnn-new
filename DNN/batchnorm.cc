%%writefile 1.cc
#include "batchnorm.h"
#include "cudnn_utility.h"

BatchNormalization::BatchNormalization(int batch, int channel, int height, int width)
    : batch(batch), channel(channel), height(height), width(width) {}

void BatchNormalization::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if (HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device output memory deallocation error\n");
  }

  cudaStatus = cudaFree(scale);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");   
  }

  cudaStatus = cudaFree(offset);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");   
  }

  cudaStatus = cudaFree(dscale);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");   
  }

  cudaStatus = cudaFree(doffset);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");   
  }

  cudaStatus = cudaFree(running_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n"); 
  }

  cudaStatus = cudaFree(running_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");  
  }

  cudaStatus = cudaFree(saved_mean);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
  }

  cudaStatus = cudaFree(saved_inv_var);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n"); 
  }

  cudaStatus = cudaFree(workspace);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");  
  }

  cudaStatus = cudaFree(reserve_space);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");  
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(mean_descriptor);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
  }
  
  status = cudnnDestroyActivationDescriptor(activation_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy activation Descriptor\n");
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
}

int BatchNormalization::BatchNormalizationApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);
  
  HostInputTensor = new float[size];
  Util::InitializeInputTensor(HostInputTensor, size);

  int num_GPUs;
  cudaStatus = cudaGetDeviceCount(&num_GPUs);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU count error");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  std::cout << "\nFound " << num_GPUs << " GPUs." << std::endl;

  //! Setting device
  cudaStatus = cudaSetDevice(0);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU set device  error");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  int device; 
  struct cudaDeviceProp devProp;
  cudaStatus = cudaGetDevice(&device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device read  error");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaGetDeviceProperties(&devProp, device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device properties  error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  std::cout << "\nCompute capability:" << devProp.major << "." << devProp.minor << std::endl;

  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  std::cout << "\nCreated cuDNN handle" << std::endl;

  int mean_size = channel;
  int mean_size_bytes = mean_size * sizeof(float);

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE; 
  }
  
  status = cudnnSetTensor4dDescriptor(input_desc, format, dtype, batch, channel, height, width); 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE; 
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  status = cudnnSetTensor4dDescriptor(output_desc, format, dtype, batch, channel, height, width); 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! Device memory allocation for Input and Output arrays
  cudaStatus = cudaMallocManaged(&DeviceInputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  cudaStatus = cudaMallocManaged(&DeviceOutputTensor, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE; 
  }

  //! Copying Input values from host to device                 
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for Input tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nOriginal array: " << std::endl; 
  
  Util::ActivationPrint(HostInputTensor, size);
  std::cout << std::endl;
  
  float alpha[channel] = {1};
  float beta[channel] = {0.0};

  status = cudnnCreateTensorDescriptor(&mean_descriptor);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating mean descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnSetTensor4dDescriptor(mean_descriptor,format, dtype, 1, channel, 1, 1);                            
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting mean descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&scale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&offset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&dscale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaMallocManaged(&doffset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&running_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&running_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&saved_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMallocManaged(&saved_inv_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! initialize scale, offset, running_mean, running_var
  for (int i = 0; i < mean_size; i++) {
    scale[i] = 1.0;
    offset[i] = 1.0;
    running_mean[i] = 1.0;
    running_var[i] = 1.0;
  }
 
  //! Setting activation descriptor
  status = cudnnCreateActivationDescriptor(&activation_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting  activation descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  status = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
           /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
           /*xDesc=*/input_desc, /*zDesc=*/NULL, /*yDesc=*/output_desc,
           /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
           /*activationDesc=*/activation_desc,
           /*sizeInBytes=*/&workspace_size_bytes);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Workspace size error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! Getting required workspace bytes
  if (workspace_size_bytes > 0) {
      cudaStatus = cudaMalloc(&workspace, workspace_size_bytes);
      if( cudaStatus != cudaSuccess) {
        printf(" the device memory allocation failed\n");
        FreeMemory();
        return EXIT_FAILURE;  
      }
  }
  
  //! Getting reserve space size bytes for the required operation
  status = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
           /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
           /*activationDesc=*/activation_desc, /*xDesc=*/input_desc,
           /*sizeInBytes=*/&reserve_space_size_bytes);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Reservespace size error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMalloc(&reserve_space, reserve_space_size_bytes);
   if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;  
  }

  clk_start=clock();
  status = cudnnBatchNormalizationForwardTraining(
        /*handle=*/handle_,
        /*mode=*/mode,
        /**alpha=*/&one,
        /**beta=*/&zero,
        /*xDesc=*/input_desc,
        /**x=*/DeviceInputTensor,
        /*yDesc=*/output_desc,
        /**y=*/DeviceOutputTensor,
        /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
        /*bnScaleData=*/scale,
        /*bnBiasData=*/offset,
        /*exponentialAverageFactor=*/0.5,
        /*resultRunningMeanData=*/running_mean,
        /*resultRunningVarianceData=*/running_var,
        /*epsilon=*/0.001,
        /*resultSaveMean=*/saved_mean,
        /*resultSaveInvVariance=*/saved_inv_var);
    
  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Kernel execution error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  HostOutputTensor= new float[size];
  
  //! Copying data from device to host
  cudaStatus = cudaMemcpy( HostOutputTensor, DeviceOutputTensor, size_bytes, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on host for output tensor failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  double flopsCoef = 2.0;
  std::cout << "\nInput n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  //! Printing the output
  std::cout << "\nThe Output array: ";
  Util::ActivationPrint(HostOutputTensor, size);

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

  //! Reading cmd line arguments and initializing the required parameters
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
  
  }

  BatchNormalization batchnormalization(batch, channel, height, width);
  batchnormalization.BatchNormalizationApiCall();
}
