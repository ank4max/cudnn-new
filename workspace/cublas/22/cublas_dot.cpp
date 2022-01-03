#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define FIRST_ARG "length"    //for comparison and assigning the length of vectors
#define FIRST_ARG_LEN 6
#define BEGIN 1
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

//1e-9 for converting throughput in GFLOP/sec, multiplying by 2 because each multiply-add operation uses two flops and 
// then divided it by latency to get required throughput


int main ( int argc, char **argv) {

  // initializing size of vector with command line arguement
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ;

  clock_t clk_start, clk_end;
  int vector_length;
  
  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }
    
  for (int loop_count = 1; loop_count < argc; loop_count++) {
    std::string str(argv[loop_count]);
    if (!((str.substr(BEGIN, FIRST_ARG_LEN)).compare(FIRST_ARG)))
      vector_length = atoi(argv[loop_count] + FIRST_ARG_LEN + 1);
  }
  
  // pointers x and y pointing  to vectors
  float * HostVecX;             
  float * HostVecY; 
  
  // host memory allocation for vectors
  HostVecX = new float[vector_length]; 
  HostVecY = new float[vector_length]; 
  
  if (HostVecX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector x)\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector y)\n");
    return EXIT_FAILURE;
  }
  
  int index; 

  //setting up values in x and y vectors
  //using rand() function to generate random numbers converted them to float 
  // and made them less than 100.
  for (index = 0; index < vector_length; index++) {
    HostVecX[index] = (rand() % 10000 * 1.00) / 100; ;                               
  }

  for (index = 0; index < vector_length; index++) {
    HostVecY[index] = (rand() % 10000 * 1.00) / 100; 
  }
  
  // printing the initial values in vector x and vector y
  std::cout <<"\nX vector initially:\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecX[index] << " "; 
  }
  std::cout <<"\n";
  
  std::cout << "\nY vector initially :\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecY[index] << " "; 
  }
  std::cout <<"\n";
  
  // Pointers for device memory allocation
  float *DeviceVecX; 
  float *DeviceVecY; 
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecX, vector_length * sizeof (*HostVecX));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecY, vector_length * sizeof (*HostVecY));
  if( cudaStatus != cudaSuccess) {
    std::cout <<" The device memory allocation failed for Y\n";
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

  float dot_product ;

  clk_start=clock();

  // performing dot product operation and storing result in result variable
  status = cublasSdot(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

  clk_end=clock();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  //printing the final result
  std::cout << "\nDot product x.y is :  " << dot_product << "\n"; 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  //freeing device memory
  cudaStatus = cudaFree (DeviceVecX);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector x\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaFree (DeviceVecY);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector y\n";
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
