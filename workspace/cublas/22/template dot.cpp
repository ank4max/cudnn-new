#include <iostream>
#include <string.h>
#include "cublas_v2.h"

#define RANDOM (rand() % 10000 * 1.00) / 100     // to generate random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

cudaError_t cudaStatus ; 
cublasStatus_t status ; 
cublasHandle_t handle ;

clock_t clk_start, clk_end;


template<typename T>
int DotPro(int vector_length, char n)
  // pointers X and Y pointing  to vectors
  T * HostVecX;             
  T * HostVecY; 
  
  // host memory allocation for vectors
  HostVecX = new T[vector_length]; 
  HostVecY = new T[vector_length]; 
  
  if (HostVecX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector X)\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector Y)\n");
    return EXIT_FAILURE;
  }
  
  int index; 

  // setting up values in X and Y vectors
  // using RANDOM macro to generate random float numbers between 0 - 100
  for (index = 0; index < vector_length; index++) {
    HostVecX[index] = RANDOM;                               
  }

  for (index = 0; index < vector_length; index++) {
    HostVecY[index] = RANDOM; 
  }
  
  // printing the initial values in vector X and vector Y
  std::cout << "\nX vector initially:\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecX[index] << " "; 
  }
  std::cout << "\n";
  
  std::cout << "\nY vector initially :\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecY[index] << " "; 
  }
  std::cout << "\n";
  
  // Pointers for device memory allocation
  T *DeviceVecX; 
  T *DeviceVecY; 
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecX, vector_length * sizeof (*HostVecX));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecY, vector_length * sizeof (*HostVecY));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;   
  }
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  status = cublasCreate (&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector (vector_length, sizeof (*HostVecX) , HostVecX, 1, DeviceVecX, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for X on gpu\n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetVector (vector_length, sizeof (*HostVecY), HostVecY, 1, DeviceVecY, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for Y on gpu\n");
    return EXIT_FAILURE;
  }

  T dot_product;
  if (n=='s') {
    std::cout<<" using sdot api\n";
    clk_start = clock();

    // performing dot product operation and storing result in dot_product variable
    status = cublasSdot(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

    clk_end = clock();
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! sdot kernel execution error\n");
      return EXIT_FAILURE;
    }
  }
  
  else if(n=='z') {
      
    clk_start = clock();

    // performing dot product operation and storing result in dot_product variable
    status = cublasDdot(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

    clk_end = clock();
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error\n");
      return EXIT_FAILURE;
    }
    
  }
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  //freeing device memory
  cudaStatus = cudaFree (DeviceVecX);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector X\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaFree (DeviceVecY);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector Y\n";
    return EXIT_FAILURE;
  }
  
  //destroying cublas context and freeing host memory
  status = cublasDestroy (handle);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize handle");
    return EXIT_FAILURE;
  }
  
  delete[] HostVecX; 
  delete[] HostVecY; 

  return EXIT_SUCCESS ;
}

  
  



int main ( int argc, char **argv) {
  // initializing size of vector with command line arguement
  
  int vector_length;
  char n;
  
  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;
    
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
    else if (!(cmd_argument.compare("-mode")))
      n = *(argv[loop_count + 1]);
      
  }
  
  if(n=='s') {
   std::cout << "using float templatefor dot function'\n";
   dot<float>(vector_length);
  }
  
  else if(n=='d') {
    std::cout<< using double template for dot function\n";
    dot<double>(vector_length);
  }
 
  return 0;
}
