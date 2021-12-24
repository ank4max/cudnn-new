

# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string.h>


char* Substr(char* InputArr, int begin, int len)
{
    char* ResultStr = new char[len + 1];
    for (int i = 0; i < len; i++)
        ResultStr[i] = *(InputArr + begin + i);
    ResultStr[len] = 0;
    return ResultStr;
}

int main (int argc, char **argv) {
  // reading cmd line arguments
  clock_t start, end;
  int x_len, y_len;
  float scalar_const;
  

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 4; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      x_len = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      y_len = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 9), "const_val"))
      scalar_const = atof(argv[i] + 10);

    
  }
  
  // length of vectorA and vectorB should be same
  if(x_len != y_len) {
      return EXIT_FAILURE;
  }
  
  // creating cublas handle
  cudaError_t cudaStatus ;
  cublasStatus_t status ;
  cublasHandle_t handle ;
  status = cublasCreate(& handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // allocating memory for vectors on host
  float *HostVecX;
  float *HostVecY;
  HostVecX = (float *) malloc(x_len * sizeof (*HostVecX));
  HostVecY = (float *) malloc(y_len * sizeof (*HostVecY));

  // setting up values in vectors
  for (int it = 0; it < x_len; it++) {
    HostVecX[it] = (float) (rand() % 10000) / 100;
  }
  for (int it = 0; it < y_len; it++) {
    HostVecY[it] = (float) (rand() % 10000) / 100;
  }

  printf ("\nOriginal vector x:\n");
  for (int it = 0; it < x_len; it++) {
    printf("%2.0f, ", HostVecX[it]);
  }
  printf ("\n");
  printf ("Original vector y:\n");
  for (int it = 0; it < y_len; it++) {
    printf ("%2.0f, ", HostVecY[it]);
  }
  printf ("\n\n");

  // using cudamalloc for allocating memory on device
  float * DeviceVecX;
  float * DeviceVecY;
  cudaStatus = cudaMalloc(( void **)& DeviceVecX, x_len * sizeof (*HostVecX));
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
    
  cudaStatus = cudaMalloc(( void **)& DeviceVecY, y_len * sizeof (*HostVecY));
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  // setting values of matrices on device
  status = cublasSetVector(x_len, sizeof (*HostVecX), HostVecX, 1, DeviceVecX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set up values in device vector X\n");
    return EXIT_FAILURE;
  }
    
  status = cublasSetVector(y_len, sizeof (*HostVecY), HostVecY, 1, DeviceVecY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to set up values in device vector Y\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();

  // performing saxpy operation
  status = cublasSaxpy(handle, x_len, &scalar_const, DeviceVecX, 1, DeviceVecY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  // end variable to store time
  end = clock();

  // getting the final output
  status = cublasGetVector(y_len, sizeof(float), DeviceVecY, 1, HostVecY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector Y\n");
    return EXIT_FAILURE;
  }

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int it = 0; it < y_len; it++) {
    printf ("%2.0f, ", HostVecY[it]);
  }
  printf ("\n\n");

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // free device memory
  cudaStatus = cudaFree(DeviceVecX);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceVecY);
  if( cudaStatus != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }

  // destroying cublas handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize");
    return EXIT_FAILURE;
  }

  // freeing host memory
  free(HostVecX);
  free(HostVecY);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
