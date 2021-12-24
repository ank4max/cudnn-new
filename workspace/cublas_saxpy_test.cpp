

# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string.h>

char* Substr(char* InputArr, int begin, int len)
{
    char* ResStr = new char[len + 1];
    for (int i = 0; i < len; i++)
        ResStr[i] = *(InputArr + begin + i);
    ResStr[len] = 0;
    return ResStr;
}

int main (int argc, char **argv) {
  // reading cmd line arguments
  clock_t start, end;
  int len_a, len_b;
  float scalConst;
  

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 4; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      len_a = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      len_b = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 9), "const_val"))
      scalConst = atof(argv[i] + 10);
  }
  
  // length of vectorA and vectorB should be same
  if(len_a != len_b) {
      return EXIT_FAILURE;
  }
  
  // creating cublas handle
  cudaError_t cudaStat ;
  cublasStatus_t stat ;
  cublasHandle_t handle ;
  stat = cublasCreate(& handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // allocating memory for vectors on host
  float *HostVecA;
  float *HostVecB;
  HostVecA = (float *) malloc(len_a * sizeof (*HostVecA));
  HostVecB = (float *) malloc(len_b * sizeof (*HostVecB));

  // setting up values in vectors
  for (int j = 0; j < len_a; j++) {
    HostVecA[j] = (float) (rand() % 10000) / 100;
  }
  for (int j = 0; j < len_b; j++) {
    HostVecB[j] = (float) (rand() % 10000) / 100;
  }

  printf ("\nOriginal vector x:\n");
  for (int j = 0; j < len_a; j++) {
    printf("%2.0f, ", HostVecA[j]);
  }
  printf ("\n");
  printf ("Original vector y:\n");
  for (int j = 0; j < len_b; j++) {
    printf ("%2.0f, ", HostVecB[j]);
  }
  printf ("\n\n");

  // using cudamalloc for allocating memory on device
  float * DevVecA;
  float * DevVecB;
  cudaStat = cudaMalloc(( void **)& DevVecA, len_a * sizeof (*HostVecA));
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
    
  cudaStat = cudaMalloc(( void **)& DevVecB, len_b * sizeof (*HostVecB));
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  // setting values of matrices on device
  stat = cublasSetVector(len_a, sizeof (*HostVecA), HostVecA, 1, DevVecA, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set up values in device vector A\n");
    return EXIT_FAILURE;
  }
    
  stat = cublasSetVector(len_b, sizeof (*HostVecB), HostVecB, 1, DevVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to set up values in device vector B\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();

  // performing saxpy operation
  stat = cublasSaxpy(handle, len_a, &scalConst, DevVecA, 1, DevVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  // end variable to store time
  end = clock();

  // getting the final output
  stat = cublasGetVector(len_b, sizeof(float), DevVecB, 1, HostVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector B\n");
    return EXIT_FAILURE;
  }

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int j = 0; j < len_b; j++) {
    printf ("%2.0f, ", HostVecB[j]);
  }
  printf ("\n\n");

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // free device memory
  cudaStat = cudaFree(DevVecA);
  if( cudaStat != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStat = cudaFree(DevVecB);
  if( cudaStat != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }

  // destroying cublas handle
  stat = cublasDestroy(handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize");
    return EXIT_FAILURE;
  }

  // freeing host memory
  free(HostVecA);
  free(HostVecB);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}


 
