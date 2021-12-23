/**
 * Copyright 2021-2022 Enflame. All Rights Reserved.
 *
 * @file    cublas_matrixMultiplication_test.cpp
 * @brief   Benchmarking Tests for cublas matrix multiplication API
 *
 * @author  ashish(CAI)
 * @date    2021-12-17
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */

#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include <iostream>
#include<string.h>
#include <time.h>

#define index(i,j,ld) (((j)*(ld))+(i))

void PrintMat(float*P, int uWP, int uHP) {
  int i, j;
  for (i = 0; i < uHP; i++) {
    printf("\n");
    for (j = 0; j < uWP; j++) {
      printf("%f ", P[index(i, j, uHP)]);
    }
  }
  printf("\n\n");
}

char* Substr(char* cInputArr, int begin, int nLen)
{
    char* pcResStr = new char[nLen + 1];
    for (int i = 0; i < nLen; i++)
        pcResStr[i] = *(cInputArr + nBegin + i);
    pcResStr[nLen] = 0;
    return pcResStr;
}

int  main(int argc, char** argv) {

  cublasStatus status;
  int i, j;
  clock_t start, end;

  // initializing cublas library
  status = cublasInit();
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize library\n");
    return EXIT_FAILURE;
  }
  // Reading dimensions of matrices
  int nRowA, nColA, nRowB, nColB, nRowC, nColC;

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 5; i++) {
        int len = sizeof(argv[i]);
        if (!strcmp(Substr(argv[i], 1, 4), "rowA"))
          nRowA = atoi(argv[i] + 5);
        else if (!strcmp(Substr(argv[i], 1, 4), "colA"))
          nColA = atoi(argv[i] + 5);
        else if (!strcmp(Substr(argv[i], 1, 4), "rowB"))
          nRowB = atoi(argv[i] + 5);
        else if (!strcmp(Substr(argv[i], 1, 4), "colB"))
          nColB = atoi(argv[i] + 5);
  }
  nRowC =  nRowA;
  nColC =  nColB ;
  
  // allocating memory for matrices on host
  float *pfMatrixA = (float*) malloc(nRowA * nColA * sizeof(float));
  float *pfMatrixB = (float*) malloc(nRowB * nColB * sizeof(float));
  float *pfMatrixC = (float*) malloc(nRowC * nColC * sizeof(float));

  if (pfMatrixA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (pfMatrixB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }
  if (pfMatrixC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  for (i = 0; i < nRowA; i++) {
    for (j = 0; j < nColA; j++) {
      pfMatrixA[index(i, j, nRowA)] = (rand() % 10000 * 1.00) / 100;
    }
  }
  for (i = 0; i < nRowB; i++) {
    for (j = 0; j < nColB; j++) {
      pfMatrixB[index(i, j, nRowB)] = (rand() % 10000 * 1.00) / 100;
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* pfDevMatA;
  float* pfDevMatB;
  float* pfDevMatC;
  status = cublasAlloc(nRowA * nColA, sizeof(float), (void**)& pfDevMatA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(nRowB * nColB, sizeof(float), (void**)& pfDevMatB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(nRowC * nColC, sizeof(float), (void**)& pfDevMatC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(nRowA, nColA, sizeof(float), pfMatrixA, nRowA, pfDevMatA, nRowA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(nRowB, nColB, sizeof(float), pfMatrixB, nRowB, pfDevMatB, nRowB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();
  

  // performing matrix multiplication
  cublasSgemm('n', 'n', nRowA, nColB, nColA, 1, pfDevMatA, nRowA, pfDevMatB, nRowB, 0, pfDevMatC, nColC);

  // end variable to store time
  end = clock();

  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  // storing the final result from device matrix to host matrix
  cublasGetMatrix(nRowC, nColC, sizeof(float), pfDevMatC, nRowC, pfMatrixC, nColC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device read error (A)\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  printf("\nMatriz A:\n");
  printMat(pfMatrixA, nColA, nRowA);
  printf("\nMatriz B:\n");
  printMat(pfMatrixB, nColB, nRowB);
  printf("\nMatriz C:\n");
  printMat(pfMatrixC, nColC, nRowC);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end-start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // freeing host memory
  free(pfMatrixA);
  free(pfMatrixB);
  free(pfMatrixC);

  // freeing device memory
  status = cublasFree(pfDevMatA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixA)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(pfDevMatB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixB)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(pfDevMatC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixC)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (matrixA)\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
