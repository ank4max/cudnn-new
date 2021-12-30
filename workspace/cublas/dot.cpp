# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <time.h>
# include <iostream>

char* SubStr(char* InputArr, int begin, int len) {
  char* ResultStr = new char[len + 1];
  for (int i = 0; i < len; i++) {
    ResultStr[i] = *(InputArr + begin + i);
  }
  ResultStr[len] = 0;
  return ResultStr;
}


int main ( int argc,char **argv ) {

  //initializing size of vector with command line arguement
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ;
  clock_t start, end;
  int x_len, y_len;
  
  for (int arg = 0;arg < argc; arg++) {
    std::cout << argv[arg] << std::endl;
  }
    
  for (int arg = 1; arg < 3; arg++) {
    if (!strcmp(SubStr(argv[arg], 1, 5), "x_len"))
      x_len = atoi(argv[arg] + 6);
    else if (!strcmp(SubStr(argv[arg], 1, 5), "y_len"))
      y_len = atoi(argv[arg] + 6);
  }

  if(x_len != y_len) {
    return EXIT_FAILURE ;
  }
 
    
  int index; 
  
  //pointers x and y pointing  to vectors
  float * HostVecX;             
  float * HostVecY; 
  
  //host memory allocation for vectors
  HostVecX = (float *) malloc (x_len* sizeof (*HostVecX)); 
  HostVecY = (float *) malloc (y_len* sizeof (*HostVecY)); 
  
  if (HostVecX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector x )\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector y )\n");
    return EXIT_FAILURE;
  }
  
  //setting up values in x and y vectors
  for (index = 0; index < x_len; index++) {
    HostVecX[index] = (float)index;                               
  }

  for (index = 0; index < y_len; index++) {
    HostVecY[index] = (float)index; 
  }
  
  //printing the initial values in vector x and vector y
  std::cout <<"X vector initially:\n";
  for (index = 0; index < x_len; index++) {
    std::cout << HostVecX[index] << " "; 
  }
  std::cout << "\n";
  
  std::cout << "Y vector initially :\n";
  for (index = 0; index < y_len; index++) {
    std::cout << HostVecY[index] << " "; 
  }
  std::cout <<"\n";
  
  // Pointers for device memory allocation
  float * DeviceVecX; 
  float * DeviceVecY; 
  
  cudaStatus = cudaMalloc ((void **)& DeviceVecX, x_len * sizeof (*HostVecX));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **)& DeviceVecY, y_len * sizeof (*HostVecY));
  
  if( cudaStatus != cudaSuccess) {
    std::cout <<" The device memory allocation failed for Y\n";
    return EXIT_FAILURE;   
  }
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  status = cublasCreate (& handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector (x_len, sizeof (*HostVecX) , HostVecX, 1, DeviceVecX, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for X on gpu\n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetVector (y_len, sizeof (*HostVecY), HostVecY, 1, DeviceVecY, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for Y on gpu\n");
    return EXIT_FAILURE;
  }

  float result ;
  // performing dot product operation and storing result in result variable
  start=clock();
  status = cublasSdot(handle, x_len, DeviceVecX, 1, DeviceVecY, 1, &result);
  end=clock();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  //printing the final result
  std::cout << "Dot product x.y is :  ";
  std::cout << result << "\n"; 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  
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
  
  free (HostVecX); 
  free (HostVecY); 
  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// dot product x.y: // x.y=
// 55 // 1*1+2*2+3*3+4*4+5*5
