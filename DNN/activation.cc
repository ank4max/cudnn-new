%%writefile me.cc
#include "activation.h"
#include "cudnn_utility.h"

Activation::Activation(int batch, int channel, int height, int width, char* activation_mode)
    : batch(batch), channel(channel), height(height), width(width),
      activation_mode(activation_mode) {}

void Activation::FreeMemory() {
    
  if(output)
  {
      delete[] output;
  }  
  
  cudaStatus = cudaFree(InputDeviceArray);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }

  cudaStatus = cudaFree(OutputDeviceArray);
  if (cudaStatus != cudaSuccess) {
    printf("Device input memory deallocation error\n");
  }


  status = cudnnDestroyTensorDescriptor(x_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }

  status = cudnnDestroyTensorDescriptor(y_desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
  }


  status = cudnnDestroy(handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("Unable to uninitialize handle\n");
  }
}

int Activation::ActivationApiCall() {
  int size = batch * channel * height * width;
  int size_bytes = size * sizeof(float);

  int InputData[size];
  for (int i = 0; i < size; i++) {
    InputData[i] = rand() % 10;
  }
 
  int num_GPUs;

  cudaStatus = cudaGetDeviceCount(&num_GPUs);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU count error");
    FreeMemory();
    return EXIT_FAILURE;  
  }
  
  std::cout << "Found " << num_GPUs << " GPUs." << std::endl;

  //setting device
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
  std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;


  status = cudnnCreate(&handle_);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }
  std::cout << "Created cuDNN handle" << std::endl;


  status = cudnnCreateTensorDescriptor(&x_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }
  
  status = cudnnSetTensor4dDescriptor(x_desc, format, dtype, batch, channel, height, width);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }

  status = cudnnCreateTensorDescriptor(&y_desc);
  if(status != CUDNN_STATUS_SUCCESS) {
    printf(" Creating tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }
  
  status = cudnnSetTensor4dDescriptor(y_desc, format, dtype, batch, channel, height, width);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Setting tensor descriptor error\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }


  cudaStatus = cudaMallocManaged(&InputDeviceArray, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE;
   
  }

  
  cudaStatus = cudaMallocManaged(&OutputDeviceArray, size_bytes);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    FreeMemory();
    return EXIT_FAILURE; 
  }

                    
  for (int i = 0; i < size; i++) {
    InputDeviceArray[i] = InputData[i] * 1.00f;
  }

  std::cout << "Original array: "; 
  for (int i = 0; i < size; i++) {
    std::cout << InputDeviceArray[i] << " ";
  }
  std::cout << std::endl;
  float alpha[channel] = {1};
  float beta[channel] = {0.0};


// Initializing activation mode 
  if (activation_mode == "tanh") {
    mode = CUDNN_ACTIVATION_TANH;
  }
  else if (activation_mode == "sigmoid") {
    mode = CUDNN_ACTIVATION_SIGMOID;
  }
  else if (activation_mode == "relu") {
    mode = CUDNN_ACTIVATION_RELU;
  }

 
  prop = CUDNN_NOT_PROPAGATE_NAN;
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


  clk_start=clock();
  status = cudnnActivationForward(handle_,
                                  activation,
                                  alpha,
                                  x_desc,
                                  InputDeviceArray,
                                  beta,
                                  y_desc,
                                  OutputDeviceArray 
                                  ); 
  clk_stop=clock();

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" kernel execution error\n");
    return EXIT_FAILURE;   
  }
  
  output= new float[size];

  cudaStatus = cudaMemcpy(output, OutputDeviceArray, size_bytes, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on host for output tensor failed\n");
    return EXIT_FAILURE;
  }


  double flopsCoef = 2.0;

  std::cout << "Input n*c*h*w: " << size <<
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;
 
  std::cout << "New array: ";
  for(int i=0;i<size;i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;



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

  //! reading cmd line arguments and initializing the required parameters
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
