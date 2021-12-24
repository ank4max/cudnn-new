# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <time.h>
# include <iostream>

char* Substr(char* InputArr, int begin, int len) {
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
  
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
    
  for (int i = 1; i < 3; i++) {
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      x_len = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      y_len = atoi(argv[i] + 5);
  }

  if(x_len != y_len) {
    return EXIT_FAILURE ;
  }
 
    
  int it; 
  
  //pointers x and y pointing  to vectors
  float * HostVecX;             
  float * HostVecY; 
  
  //host memory allocation for vectors
  HostVecX = ( float *) malloc ( x_len* sizeof (*HostVecX)); 
  HostVecY = ( float *) malloc ( y_len* sizeof (*HostVecY)); 
  
  if (HostVecX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector x )\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (vector y )\n");
    return EXIT_FAILURE;
  }
  


  //setting up values in x and y vectors
  for(it = 0;it < x_len; it++) {
    HostVecX[it] = ( float )it; // x={0 ,1 ,2 ,3 ,4 ,5}
  }

  for (it = 0; it < y_len ; it++) {
    HostVecY[it] = ( float )it; 
  }
  
  //printing the initial values in vector x and vector y
  printf ("x:\n");
  for (it = 0; it < x_len; it++) {
    printf (" %2.0f,",HostVecX[it]); 
  }
  printf ("\n");
  
   printf ("y:\n");
  for (it = 0; it < y_len; it++) {
    printf (" %2.0f,",HostVecY[it]); 
  }
  printf ("\n");
  
  // Pointers for device memory allocation
  float * DeviceVecX; 
  float * DeviceVecY; 
  
  cudaStatus = cudaMalloc (( void **)& DeviceVecX, x_len* sizeof (*HostVecX));
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc (( void **)& DeviceVecY, y_len* sizeof (*HostVecY));
  
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  status = cublasCreate (& handle );
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
  stat=cublasSdot(handle, x_len, DeviceVecX, 1, DeviceVecY, 1, &result);
  end=clock();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  //printing the final result
  printf ("dot product x.y:\n");
  printf (" %7.0f",result ); 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  
  //freeing device memory
  cudaStatus = cudaFree (DeviceVecX );
  if( cudaStatus != cudaSuccess) {
    printf(" Memory free error on device for vector x\n");
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaFree (DeviceVecY );
  if( cudaStatus != cudaSuccess) {
    printf(" Memory free error on device for vector y\n");
    return EXIT_FAILURE;
  }
  
  //destroying cublas context and freeing host memory
  status = cublasDestroy ( handle );
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (matrixA)\n");
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
