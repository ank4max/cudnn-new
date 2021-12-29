#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>
#define index(i,j,ld) (((j)*(ld))+(i))

void PrintMat(float* PrintMatrix, int col, int row) {
  int i, j;
  for (i = 0; i < row; i++) {
    std::cout << "\n";
    for (j = 0; j < col; j++) {
      std::cout << PrintMatrix[index(i, j, row)] << " ";
    }
  }
  std::cout << "\n";
}


char* SubStr(char* InputArr, int begin, int len) {
  char* ResultStr = new char[len + 1];
  for (int i = 0; i < len; i++) {
    ResultStr[i] = *(InputArr + begin + i);
  }
  ResultStr[len] = 0;
  return ResultStr;
}

int main (int argc, char **argv) {
  //variables for dimension of matrices
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha, beta;
  //status variable declaration
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ;
  clock_t start, end;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 7; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(SubStr(argv[i], 1, 5), "x_row"))
      x_row = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "x_col"))
      x_col = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "y_row"))
      y_row = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "y_col"))
      y_col = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "alpha"))
      alpha = atof(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 4), "beta"))
      beta = atof(argv[i] + 5);
  }
  
  if (x_row != x_col) {
    return EXIT_FAILURE;
  }
  
  if (x_row != y_row) {
    return EXIT_FAILURE;
  }
  
  z_row = y_row;
  z_col = y_col;
  
  int i,j; // i-row ind. , j- column ind.
  float * HostMatX; // mxm matrix a on the host
  float * HostMatY; // mxn matrix b on the host
  float * HostMatZ; // mxn matrix c on the host
  
  HostMatX = (float *) malloc (x_row * x_col * sizeof (float)); // host memory for a
  HostMatY = (float *) malloc (y_row * y_col * sizeof (float)); // host memory for b
  HostMatZ = (float *) malloc (z_row * z_col * sizeof (float)); // host memory for c
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
  
  // define the lower triangle of an mxm symmetric matrix x in
  // lower mode column by column
  int ind =11; // a:
  for (j = 0; j < x_col; j++) {                             // 11
    for (i = 0; i < x_row; i++) {                         // 12 ,17
      if(i >=j) {                                               // 13 ,18 ,22
        HostMatX[index(i, j, x_row)] = (float)ind ++;          // 14 ,19 ,23 ,26
      }                                                        // 15 ,20 ,24 ,27 ,29
    }                                                       // 16 ,21 ,25 ,28 ,30 ,31
  }
  // print the lower triangle of a row by row
  std::cout << " lower triangle of x:\n";
  for (i = 0; i < x_row; i++) {
    for (j = 0; j < x_col; j++) {
      if (i >=j) {
        std::cout << HostMatX[index(i, j, x_row)];
      }
      
    }
    std::cout<<"\n";
  }
  
  // define mxn matrices y column by column
  ind =11; 
  for (j = 0; j < y_col; j++) {                // 11 ,17 ,23 ,29
    for (i = 0; i < y_row; i++) {                        // 12 ,18 ,24 ,30
      HostMatY[index(i, j, y_row)] = (float)ind;         // 13 ,19 ,25 ,31                                                        14 ,20 ,26 ,32
      ind ++; // 15 ,21 ,27 ,33
    } // 16 ,22 ,28 ,34
  }
  
  // define mxn matrices z column by column
  ind =11; 
  for (j = 0; j < z_col; j++) {                
    for (i = 0; i < z_row; i++) {                        
      HostMatZ[index(i, j, z_row)] = (float)ind;                                                                
      ind ++;                                                  // 15 ,21 ,27 ,33
    }                                                     // 16 ,22 ,28 ,34
  }
  
  
  // print y row by row
  // print z row by row
  std::cout << "\nMatriz Y:\n";
  PrintMat(HostMatY, y_col, y_row);
  std::cout << "\nMatriz Z:\n";
  PrintMat(HostMatZ, z_col, z_row);
 
  // on the device
  float *DeviceMatX; // d_a - a on the device
  float *DeviceMatY; // d_b - b on the device
  float *DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc((void **)& DeviceMatX, x_row * x_col * sizeof (*HostMatX)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for X\n");
    return EXIT_FAILURE;
  }
  // memory alloc for y
  cudaStatus = cudaMalloc((void **)& DeviceMatY, y_row * y_col * sizeof (*HostMatY)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Y\n");
    return EXIT_FAILURE;
  }
  // memory alloc for z
  cudaStatus = cudaMalloc((void **)& DeviceMatZ, z_row * z_col * sizeof (*HostMatZ)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Z\n");
    return EXIT_FAILURE;
  }
  
  status = cublasCreate (& handle); // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  // copy matrices from the host to the device
  status = cublasSetMatrix (x_row, x_col, sizeof (*HostMatX), HostMatX, x_row, DeviceMatX, x_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (y_row, y_col, sizeof (*HostMatY), HostMatY, y_row, DeviceMatY, y_row); //b -> d_b
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed\n");
    return EXIT_FAILURE;
  }
 
  status = cublasSetMatrix (z_row, z_col, sizeof (*HostMatZ), HostMatZ, z_row, DeviceMatZ, z_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed\n");
    return EXIT_FAILURE;
  }
  
 
  // symmetric matrix - matrix multiplication :
  // d_c = al*d_a *d_b + bet *d_c ; d_a - mxm symmetric matrix ;
  // d_b ,d_c - mxn general matrices ; al ,bet - scalars;
  
  start = clock();
  
  status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
  y_row, y_col, &alpha, DeviceMatX, x_row, DeviceMatY, y_row, &beta, DeviceMatZ, z_row);
  
  end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  status = cublasGetMatrix (z_row, z_col, sizeof (*HostMatZ), DeviceMatZ, z_row, HostMatZ, z_row); // d_c -> c
  
  std::cout << "\nMatriz Z after Symm operation is:\n";
  PrintMat(HostMatZ, z_col, z_row);
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";
  
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for X\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for Y\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatZ); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for Z\n";
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy (handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  } 
  
  free (HostMatX); // free host memory
  free (HostMatY); // free host memory
  free (HostMatZ); // free host memory
  return EXIT_SUCCESS ;
}
// lower triangle of a:
// 11
// 12 17
// 13 18 22
// 14 19 23 26
// 15 20 24 27 29
// 16 21 25 28 30 31
// b(=c):
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34
// c after Ssymm :
// 1122 1614 2106 2598
// 1484 2132 2780 3428
// 1740 2496 3252 4008 // c=al*a*b+bet *c
// 1912 2740 3568 4396
// 2025 2901 3777 4653
// 2107 3019 3931 4843
