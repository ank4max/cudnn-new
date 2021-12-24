/**
 * Copyright 2021-2022 Enflame. All Rights Reserved.
 *
 * @file    cublas_saxpy_test.cpp
 * @brief   Benchmarking Tests for cublas Saxpy API
 *
 * @author  ashish(CAI)
 * @date    2021-12-17
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */


# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string.h>

char* Substr(char* cInputArr, int nBegin, int nLen)
{
    char* pcResStr = new char[nLen + 1];
    for (int i = 0; i < nLen; i++)
        pcResStr[i] = *(cInputArr + nBegin + i);
    pcResStr[nLen] = 0;
    return pcResStr;
}

int main (int argc, char **argv) {
  // reading cmd line arguments
  clock_t start, end;
  int nLenA, nLenB;
  float fScalConst;
  

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 4; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      nLenA = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      nLenB = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 9), "const_val"))
      fScalConst = atof(argv[i] + 10);
  }
  
  // length of vectorA and vectorB should be same
  if(nLenA != nLenB) {
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
  float *pfHostVecA;
  float *pfHostVecB;
  pfHostVecA = (float *) malloc(nLenA * sizeof (*pfHostVecA));
  pfHostVecB = (float *) malloc(nLenB * sizeof (*pfHostVecB));

  // setting up values in vectors
  for (int j = 0; j < nLenA; j++) {
    pfHostVecA[j] = (float) (rand() % 10000) / 100;
  }
  for (int j = 0; j < lenB; j++) {
    pfHostVecB[j] = (float) (rand() % 10000) / 100;
  }

  printf ("\nOriginal vector x:\n");
  for (int j = 0; j < nLenA; j++) {
    printf("%2.0f, ", pfHostVecA[j]);
  }
  printf ("\n");
  printf ("Original vector y:\n");
  for (int j = 0; j < nLenB; j++) {
    printf ("%2.0f, ", pfHostVecB[j]);
  }
  printf ("\n\n");

  // using cudamalloc for allocating memory on device
  float * pfDevVecA;
  float * pfDevVecB;
  cudaStat = cudaMalloc(( void **)& pfDevVecA, nLenA * sizeof (*pfHostVecA));
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
    
  cudaStat = cudaMalloc(( void **)& pfDevVecB, nLenB * sizeof (*pfHostVecB));
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
  // setting values of matrices on device
  stat = cublasSetVector(nLenA, sizeof (*pfHostVecA), pfHostVecA, 1, pfDevVecA, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set up values in device vector A\n");
    return EXIT_FAILURE;
  }
    
  stat = cublasSetVector(nLenB, sizeof (*pfHostVecB), pfHostVecB, 1, pfDevVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to set up values in device vector B\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();

  // performing saxpy operation
  stat = cublasSaxpy(handle, nLenA, &fScalConst, pfDevVecA, 1, pfDevVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  // end variable to store time
  end = clock();

  // getting the final output
  stat = cublasGetVector(nLenB, sizeof(float), pfDevVecB, 1, pfHostVecB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector B\n");
    return EXIT_FAILURE;
  }

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int j = 0; j < nLenB; j++) {
    printf ("%2.0f, ", pfHostVecB[j]);
  }
  printf ("\n\n");

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // free device memory
  cudaFree(vectorAA);
  cudaFree(vectorBB);

  // destroying cublas handle
  stat = cublasDestroy(handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize");
    return EXIT_FAILURE;
  }

  // freeing host memory
  free(vectorA);
  free(vectorB);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}


 
