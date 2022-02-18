%%writefile 5.cc
#include "activation.h"
#include "cudnn_utility.h"

Activation::Activation(int batch, int channel, int height, int width, char* activation_mode)
    : batch(batch), channel(channel), height(height), width(width),
      activation_mode(activation_mode) {}

void Activation::FreeMemory() {

  if(HostInputTensor) {
    delete[] HostInputTensor;
    HostInputTensor = nullptr;
  }

  if(HostOutputTensor) {
    delete[] HostOutputTensor;
    HostOutputTensor = nullptr;
  }  
 
  cudaStatus = cudaFree(DeviceInputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }

  cudaStatus = cudaFree(DeviceOutputTensor);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(output_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
  }

  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
}

int Activation::ActivationApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);
  
  //! Initializing input data
  HostInputTensor = new float[size];
  HostOutputTensor= new float[size];

  Util::InitializeActivationTensor(HostInputTensor, size);

  //! Printing initial array before activation
  std::cout << "\nOriginal array: "; 
  Util::PrintTensor(HostInputTensor, batch, channel, height, width);
  std::cout << std::endl;
 
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
    printf(" \nGPU set device  error\n");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  int device; 
  struct cudaDeviceProp devProp;
  cudaStatus = cudaGetDevice(&device);
  if( cudaStatus != cudaSuccess) {
    printf(" \nDevice read  error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaGetDeviceProperties(&devProp, device);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice properties  error\n");
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

  //! Device memory allocation for Input and Output Arrays
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
    
  float alpha[channel] = {1};
  float beta[channel] = {0.0};

  //! Initializing activation mode 
  if (activation_mode == "tanh") {
    mode = CUDNN_ACTIVATION_TANH;
  }
  else if (activation_mode == "sigmoid") {
    mode = CUDNN_ACTIVATION_SIGMOID;
  }
  else if (activation_mode == "relu") {
    mode = CUDNN_ACTIVATION_RELU;
  }

  //! Setting activation descriptor
  prop = CUDNN_NOT_PROPAGATE_NAN;
  status = cudnnCreateActivationDescriptor(&activation_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating activation descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }
  
  status = cudnnSetActivationDescriptor(activation_desc, mode, prop, 0.0f);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf("Setting activation  descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  }

  //! API call
  clk_start=clock();
  status = cudnnActivationForward(handle_,
                                  activation_desc,
                                  alpha,
                                  input_desc,
                                  DeviceInputTensor,
                                  beta,
                                  output_desc,
                                  DeviceOutputTensor 
                                  ); 
  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" kernel execution error\n");
    FreeMemory();
    return EXIT_FAILURE;   
  } 
  
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
 
  std::cout << "\nNew array: ";
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

  int batch, channel, height, width;
  char* activation_mode;

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

    else if (!(cmd_argument.compare("-activation_mode")))
      activation_mode = argv[loop_count + 1];
  }

  Activation activation(batch, channel, height, width, activation_mode);
  activation.ActivationApiCall();
}
