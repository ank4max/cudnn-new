#include "Dropout.h"

//print function

void printArr3D(float * arr, int arrH, int arrW, int batchSize)
{
	for(int i = 0; i < batchSize; i++)
	{
		for(int j = 0; j < arrH; j++)
		{
			for(int k = 0; k < arrW; k++)
			{
				printf("%f ", arr[i*arrH*arrW + j*arrW + k]);
			}
			printf("\n");
		}
		printf("\n");
	}
}


void GPU_PrintArr3D(float* d_arr, int arrH, int arrW, int batchSize)
{
	float* h_arr;
	h_arr = (float *) malloc(sizeof(float)*arrH*arrW*batchSize);
	cudaMemcpy(h_arr,d_arr,sizeof(float)*arrH*arrW*batchSize,cudaMemcpyDeviceToHost);
	printArr3D(h_arr,arrH,arrW,batchSize);
}

Dropout::Dropout(int batch, int channel, int height, int width, float drop_rate)
    : batch(batch), channel(channel), height(height), width(width), drop_rate(drop_rate)) {}

int Dropout::FreeMemory() {
  
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

  cudaStatus = cudaFree(d_dropout_out);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }
	
  cudaStatus = cudaFree(d_dx_dropout);
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
  cudaMallocManaged(&states, dropout_state_size);
  if( cudaStatus != cudaSuccess) {
    printf("\nDevice Memory allocation error \n");
    return EXIT_FAILURE;   
  }
  
  cudaMallocManaged(&dropout_reserve_space, dropout_reserve_size);
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
  
  //cudaMemcpy(d_input,InputData,size_bytes,cudaMemcpyHostToHost);
  printf("Input \n");
  GPU_PrintArr3D(InputData,height,width,batch * channel);
  
  //! API call
  clk_start=clock();
      
  status = cudnnDropoutForward(handle_, dropout_descriptor, dropout_in_out_descriptor, InputData, dropout_in_out_descriptor,
			       d_dropout_out, dropout_reserve_space, dropout_reserve_size);
  
  clk_stop=clock();
      
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" API faied to execute\n");
    return EXIT_FAILURE;   
  }
	
  double flopsCoef = 2.0;
  std::cout << "\nInput n*c*h*w: " << size << 
               "\nLatency: " << ((double)(clk_stop - clk_start))/CLOCKS_PER_SEC <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_stop, size) << std::endl;
  
  printf("Dropout \n");
  GPU_PrintArr3D(d_dropout_out,Height,width,batch * channel);
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

  Dropout dropout(batch, channel, height, width);
  dropout.DropoutForwardApiCall();
}
  
  
