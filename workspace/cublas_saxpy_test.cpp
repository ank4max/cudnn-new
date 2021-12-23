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
  float scalar_const;
  

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 4; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      lenA = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      lenB = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 9), "const_val"))
      scalar_const = atof(argv[i] + 10);
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
  float *vectorA;
  float *vectorB;
  vectorA = (float *) malloc(lenA * sizeof (*vectorA));
  vectorB = (float *) malloc(lenB * sizeof (*vectorB));

  // setting up values in vectors
  for (int j = 0; j < lenA; j++) {
    vectorA[j] = (float) (rand() % 10000) / 100;
  }
  for (int j = 0; j < lenB; j++) {
    vectorB[j] = (float) (rand() % 10000) / 100;
  }

  printf ("\nOriginal vector x:\n");
  for (int j = 0; j < lenA; j++) {
    printf("%2.0f, ", vectorA[j]);
  }
  printf ("\n");
  printf ("Original vector y:\n");
  for (int j = 0; j < lenB; j++) {
    printf ("%2.0f, ", vectorB[j]);
  }
  printf ("\n\n");

  // using cudamalloc for allocating memory on device
  float * vectorAA;
  float * vectorBB;
  cudaStat = cudaMalloc(( void **)& vectorAA, lenA * sizeof (*vectorA));
  cudaStat = cudaMalloc(( void **)& vectorBB, lenB * sizeof (*vectorB));

  // setting values of matrices on device
  stat = cublasSetVector(lenA, sizeof (*vectorA), vectorA, 1, vectorAA, 1);
  stat = cublasSetVector(lenB, sizeof (*vectorB), vectorB, 1, vectorBB, 1);

  // start variable to store time
  start = clock();

  // performing saxpy operation
  stat = cublasSaxpy(handle, lenA, &scalar_const, vectorAA, 1, vectorBB, 1);

  // end variable to store time
  end = clock();

  // getting the final output
  stat = cublasGetVector(lenB, sizeof(float), vectorBB, 1, vectorB, 1);

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int j = 0; j < lenB; j++) {
    printf ("%2.0f, ", vectorB[j]);
  }
  printf ("\n\n");

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // free device memory
  cudaFree(vectorAA);
  cudaFree(vectorBB);

  // destroying cublas handle
  cublasDestroy(handle);

  // freeing host memory
  free(vectorA);
  free(vectorB);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}


 
