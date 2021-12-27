
#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include <iostream>
#include<string.h>
#include <time.h>

#define index(i,j,ld) (((j)*(ld))+(i))

void PrintMat(float*PrintMatrix, int col, int row) {
  int i, j;
  for (i = 0; i < row; i++) {
    printf("\n");
    for (j = 0; j < col; j++) {
      printf("%f ", PrintMatrix[index(i, j,row)]);
    }
  }
  printf("\n\n");
}

char* Substr(char* InputArr, int begin, int len) {
  char* ResultStr = new char[len + 1];
  for (int i = 0; i < len; i++) {
    ResultStr[i] = *(InputArr + begin + i);
  }
  ResultStr[len] = 0;
  return ResultStr;
}

int  main(int argc, char** argv) {

  cublasStatus status;
  int i, j;
  clock_t start, end;

  // initializing cublas library
  status = cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize library\n");
    return EXIT_FAILURE;
  }
  // Reading dimensions of matrices
  int x_row, x_col, y_row, y_col, z_row, z_col;

  std::cout << "\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 5; i++) {
        int len = sizeof(argv[i]);
        if (!strcmp(Substr(argv[i], 1, 5), "row_x"))
          x_row = atoi(argv[i] + 6);
        else if (!strcmp(Substr(argv[i], 1, 5), "col_x"))
          x_col = atoi(argv[i] + 6);
        else if (!strcmp(Substr(argv[i], 1, 5), "row_y"))
          y_row = atoi(argv[i] + 6);
        else if (!strcmp(Substr(argv[i], 1, 5), "col_y"))
          y_col = atoi(argv[i] + 6);
  }
  z_row =  x_row;
  z_col =  y_col ;
  
  // allocating memory for matrices on host
  float *HostMatX = (float*) malloc(x_row * x_col * sizeof(float));
  float *HostMatY = (float*) malloc(y_row * y_col * sizeof(float));
  float *HostMatZ = (float*) malloc(z_row * z_col * sizeof(float));

  if (HostMatX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixZ)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  for (i = 0; i < x_row; i++) {
    for (j = 0; j < x_col; j++) {
      HostMatX[index(i, j, x_row)] = (rand() % 10000 * 1.00) / 100;
    }
  }
  for (i = 0; i < y_row; i++) {
    for (j = 0; j < y_col; j++) {
      HostMatY[index(i, j, y_row)] = (rand() % 10000 * 1.00) / 100;
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* DeviceMatX;
  float* DeviceMatY;
  float* DeviceMatZ;
  status = cublasAlloc(x_row * x_col, sizeof(float), (void**)& DeviceMatX);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (X)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(y_row * y_col, sizeof(float), (void**)& DeviceMatY);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (Y)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(z_row * z_col, sizeof(float), (void**)& DeviceMatZ);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (Z)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(x_row, x_col, sizeof(float), HostMatX, x_row, DeviceMatX, x_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (X) failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(y_row, y_col, sizeof(float), HostMatY, y_row, DeviceMatY,y_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (Y) failed\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();
  

  // performing matrix multiplication
  cublasSgemm('n', 'n', x_row, y_col, x_col, 1, DeviceMatX, x_row, DeviceMatY, y_row, 0, DeviceMatZ, z_col);

  // end variable to store time
  end = clock();

  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  // storing the final result from device matrix to host matrix
  cublasGetMatrix(z_row, z_col, sizeof(float), DeviceMatZ, z_row, HostMatZ, z_col);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device read error (A)\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  printf("\nMatriz X:\n");
  PrintMat(HostMatX, x_col, x_row);
  printf("\nMatriz Y:\n");
  PrintMat(HostMatY, y_col, y_row);
  printf("\nMatriz Z:\n");
  PrintMat(HostMatZ, z_col, z_row);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end-start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // freeing host memory
  free(HostMatX);
  free(HostMatY);
  free(HostMatZ);

  // freeing device memory
  status = cublasFree(DeviceMatX);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixX)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(DeviceMatY);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixY)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(DeviceMatZ);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixZ)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
