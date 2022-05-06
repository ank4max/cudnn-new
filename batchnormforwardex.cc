%%writefile cu.cc
#include "cudnn_BatchNormalizationForwardTrainingEx_test.h"
#include "cudnn_utility.h"

#define ExponentialAverageFactor 0.5
#define EPSILON 0.001

BatchNormalizationForwardTrainingEx::BatchNormalizationForwardTrainingEx
   (int batch, int channel, int height, int width, char * batchnorm_mode, 
   char *activate_mode, char *norm_ops) : batch(batch), channel(channel), 
   height(height), width(width), batchnorm_mode(batchnorm_mode) 
   ,activate_mode(activate_mode), norm_ops(norm_ops){}

void BatchNormalizationForwardTrainingEx::FreeMemory() {
  if (HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if (HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }

  if (scale) {
    delete[] scale;
    scale = nullptr;
  }

  if (offset) {
    delete[] offset;
    offset = nullptr;
  }

  if (running_mean) {
    delete[] running_mean;
    running_mean = nullptr;
  }

  if (running_var) {
    delete[] running_var;
    running_var = nullptr;
  }

  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device input memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Device output memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_scale);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device scale memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_offset);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device offset memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_running_mean);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device running_mean memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_running_var);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device running_var memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_saved_mean);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device saved_mean memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(device_saved_inv_var);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device saved_inv_var memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(workspace_data);
  if( cudaStatus != cudaSuccess) {
    std::cout << " workspace_data memory deallocation error\n" << std::endl;
  }

  cudaStatus = cudaFree(reserve_data);
  if( cudaStatus != cudaSuccess) {
    std::cout << " workspace_data memory deallocation error\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy input Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy output Descriptor\n" << std::endl;
  }

  status = cudnnDestroyTensorDescriptor(mean_descriptor);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to Destroy mean Descriptor\n" << std::endl;
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Unable to uninitialize handle\n" << std::endl;
  }
}

int BatchNormalizationForwardTrainingEx::BatchNormalizationForwardTrainingExApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  int mean_size = channel;
  int mean_size_bytes = mean_size * sizeof(float);

  HostInputTensor = new float[size];
  HostOutputTensor = new float[size];

  Util::InitializeInputTensor(HostInputTensor, size);

  std::cout << "\nInput_data:" << std::endl;
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);

  // Create cudnn handle
  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Unable to initialize handle\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  std::cout << "Created cuDNN handle" << std::endl;

  status = cudnnCreateTensorDescriptor(&input_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(input_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting input tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnCreateTensorDescriptor(&output_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating output tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(output_desc, data_format, data_type,
                                      batch, channel, height, width);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting output tensor descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&DeviceInputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    std::cout << " Memory allocation on device for input tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMalloc(&DeviceOutputTensor, size_bytes);
  if(cudaStatus != cudaSuccess) {
    std::cout << " Memory allocation on device for output tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying Input values from host to device
  cudaStatus = cudaMemcpy(DeviceInputTensor, HostInputTensor, size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for Input tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * CUDNN_BATCHNORM_PER_ACTIVATION
   *    Normalization is performed per-activation. This mode is intended to be used
   *    after non-convolutional network layers. In this mode bnBias and bnScale tensor
   *    dimensions are 1xCxHxW.
   * CUDNN_BATCHNORM_SPATIAL
   *    Normalization is performed over N+spatial dimensions. This mode is intended for
   *    use after convolutional layers (where spatial invariance is desired). In this mode
   *    bnBias, bnScale tensor dimensions are 1xCx1x1.
   * CUDNN_BATCHNORM_SPATIAL_PERSISTENT
   *    This mode is similar to CUDNN_BATCHNORM_SPATIAL but it
   *    can be faster for some tasks.
   */
  if (batchnorm_mode == "per_activation") {
    bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    std::cout << "\nUsing batchnorm mode : CUDNN_BATCHNORM_PER_ACTIVATION\n";
  }

  else if (batchnorm_mode == "spatial") {
    bn_mode = CUDNN_BATCHNORM_SPATIAL;
    std::cout << "\nUsing batchnorm mode : CUDNN_BATCHNORM_SPATIAL\n";
  }
  
  else {
    bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    std::cout <<"\nUsing batchnorm mode : CUDNN_BATCHNORM_SPATIAL_PERSISTENT\n";
  }

  alpha = ALPHA_INITIAL_VALUE;
  beta= BETA_INITIAL_VALUE;

  status = cudnnCreateTensorDescriptor(&mean_descriptor);
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating mean descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetTensor4dDescriptor(mean_descriptor, data_format, data_type,
                                      1, mean_size, 1, 1);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Setting mean descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  scale = new float[mean_size];
  offset = new float[mean_size];
  running_mean = new float[mean_size];
  running_var = new float[mean_size];

  //! initialize scale, offset, running_mean, running_var
  for (int index = 0; index < mean_size; index++) {
    scale[index] = PARAM_INITIAL_VALUE;
    offset[index] = PARAM_INITIAL_VALUE;
    running_mean[index] = PARAM_INITIAL_VALUE;
    running_var[index] = PARAM_INITIAL_VALUE;
  }

  cudaStatus = cudaMalloc(&device_scale, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for scale\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_offset, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for offset\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for running_mean\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_running_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed for running_var\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_scale, scale, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_offset, offset, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_mean, running_mean, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMemcpy(device_running_var, running_var, mean_size_bytes,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on device for scale tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_mean, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc(&device_saved_inv_var, mean_size_bytes);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory allocation failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * CUDNN_ACTIVATION_SIGMOID
   *    Selects the sigmoid function.
   * CUDNN_ACTIVATION_RELU
   *    Selects the rectified linear function.
   * CUDNN_ACTIVATION_TANH
   *    Selects the hyperbolic tangent function.
   */


  //! Initializing activation mode
  if (activate_mode == "tanh") {
    activation_mode = CUDNN_ACTIVATION_TANH;
    std::cout << "Using activation mode: CUDNN_ACTIVATION_TANH" << std::endl;
  }
  else if (activate_mode == "relu") {
    activation_mode = CUDNN_ACTIVATION_RELU;
    std::cout << "Using activation mode: CUDNN_ACTIVATION_RELU" << std::endl;
  }
  else {
    activation_mode = CUDNN_ACTIVATION_SIGMOID;
    std::cout << "Using activation mode: CUDNN_ACTIVATION_SIGMOID" << std::endl;
  }

  //! Setting activation descriptor
  status = cudnnCreateActivationDescriptor(&activation_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Creating activation descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cudnnSetActivationDescriptor(activation_desc, activation_mode, propagation, RELU_CLIPPING_THREASHOLD);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << "Setting activation  descriptor error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * CUDNN_BATCHNORM_OPS_BN
   *    Only batch normalization is performed, per-activation.
   * CUDNN_BATCHNORM_OPS_BN_ACTIVATION
   *    First, the batch normalization is performed, and then the activation 
   *    is performed.
   * CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION
   *    Performs the batch normalization, then element-wise addition, 
   *    followed by the activation operation.
   */


  //! Setting mode for  normops
  if(norm_ops == "bn") {
      bn_ops = CUDNN_BATCHNORM_OPS_BN;
      std::cout << "\n Using CUDNN_BATCHNORM_OPS_BN \n";
  }

  else if(norm_ops == "bn_actiavtion") {
      bn_ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
      std::cout << "\n Using CUDNN_BATCHNORM_OPS_BN_ACTIVATION \n";
  }

  else {
      bn_ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
      std::cout << "\n Using CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION \n";
  }

  //! workspace
  size_t workspace_size;
  status = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle_, 
              bn_mode, bn_ops, input_desc, output_desc, output_desc, 
              mean_descriptor, activation_desc, &workspace_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Getting Forward Workspace size error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMalloc(&workspace_data, workspace_size);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device workspace_size memory allocation failed\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  size_t reserve_size;
  status = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle_, bn_mode, bn_ops, activation_desc, input_desc, &reserve_size);
  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Getting reserve Workspace size error\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMalloc(&reserve_data, reserve_size);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Device reserve_size memory allocation failed\n" << std::endl;  
    FreeMemory();
    return EXIT_FAILURE;   
  }


  /**
   * This API is an extension of the cudnnBatchNormalizationForwardTraining() for performing the forward batch normalization layer computation.
   * This API will trigger the new semi-persistent NHWC kernel when the following conditions are true:
   *    All tensors, namely, x, y, dz, dy, dx must be NHWC-fully packed and must be of the type CUDNN_DATA_HALF.
   *    The input parameter mode must be set to CUDNN_BATCHNORM_SPATIAL_PERSISTENT.workspace is not NULL.
   *    Before cuDNN version 8.2.0, the tensor C dimension should always be a multiple of 4. 
   *    After 8.2.0, the tensor C dimension should be a multiple of 4 only when bnOps is CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION.
   *    WorkSpaceSizeInBytes is equal to or larger than the amount required by cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize().
   *    ReserveSpaceSizeInBytes is equal to or larger than the amount required by cudnnGetBatchNormalizationTrainingExReserveSpaceSize().
   *    The content in reserveSpace stored by cudnnBatchNormalizationForwardTrainingEx() must be preserved.
   * If workspace is NULL and workSpaceSizeInBytes of zero is passed in, 
   *  this API will function exactly like the non-extended function cudnnBatchNormalizationForwardTraining().
   * This workspace is not required to be clean. Moreover, the workspace does not have to remain unchanged 
   *  between the forward and backward pass, as it is not used for passing any information.
   * This extended function can accept a *workspace pointer to the GPU workspace, and workSpaceSizeInBytes, the size of the workspace, from the user.
   * The bnOps input can be used to set this function to perform either only the batch normalization, or batch normalization followed by activation, 
   *  or batch normalization followed by element-wise addition and then activation.
   * Only 4D and 5D tensors are supported. The epsilon value has to be the same during the training, the backpropagation, and the inference.
   */


   /**
    * CUDNN_STATUS_SUCCESS - The computation was performed successfully. \n
    * CUDNN_STATUS_NOT_SUPPORTED - The function does not support the provided configuration. \n
    * CUDNN_STATUS_BAD_PARAM - At least one of the following conditions are met:
    *   One of the pointers alpha, beta, x, y, bnScaleData, bnBiasData is NULL.
    *   The number of xDesc or yDesc tensor descriptor dimensions is not within the [4,5] range (only 4D and 5D tensors are supported).
    *   bnScaleBiasMeanVarDesc dimensions are not 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for spatial, and are not 1xCxHxW for 4D and 1xCxDxHxW 
    *    for 5D for per-activation mode.
    *   Exactly one of saveMean, saveInvVariance pointers are NULL.
    *   Exactly one of resultRunningMeanData, resultRunningInvVarianceData pointers are NULL.
    *   epsilon value is less than CUDNN_BN_MIN_EPSILON.
    *   Dimensions or data types mismatch for xDesc, yDesc.
    */

  clk_start=clock();

  status = cudnnBatchNormalizationForwardTrainingEx(handle_, bn_mode, bn_ops,
                &alpha, &beta, input_desc, DeviceInputTensor, output_desc,
                DeviceOutputTensor, output_desc,
                DeviceOutputTensor, mean_descriptor, device_scale,
                device_offset, ExponentialAverageFactor, device_running_mean,
                device_running_var,EPSILON, device_saved_mean,
                device_saved_inv_var, activation_desc, workspace_data, 
                workspace_size, reserve_data, reserve_size);

  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    std::cout << " Kernel execution error\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying data from device to host
  cudaStatus = cudaMemcpy(HostOutputTensor, DeviceOutputTensor, size_bytes,
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Setting up values on host for output tensor failed\n" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nInput n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;

  //! Printing the output
  std::cout << "\nOutput_data:" << std::endl;
  Util::PrintTensor(HostOutputTensor, batch, channel, height, width);

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

  int batch, channel, height, width, status;
  char *batchnorm_mode, *activate_mode, *norm_ops;

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

    else if (!(cmd_argument.compare("-batchnorm_mode")))
      batchnorm_mode = (argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-activation_mode")))
      activate_mode = (argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-norm_ops")))
      norm_ops = (argv[loop_count + 1]);
  }

  BatchNormalizationForwardTrainingEx BatchNormalizationForwardTrainingEx(batch, channel, height, width, batchnorm_mode, activate_mode, norm_ops);
  status = BatchNormalizationForwardTrainingEx.BatchNormalizationForwardTrainingExApiCall();
  return status;
}
