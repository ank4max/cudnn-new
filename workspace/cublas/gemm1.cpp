#include<iostream>
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
      std::cout << PrintMatrix[index(i, j,row)] << " ";
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

int main (int argc, char **argv  ) {
  
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha, beta;
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 6; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(SubStr(argv[i], 1, 5), "x_row"))
      x_row = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "x_col"))
      x_col = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "y_col"))
      y_col = atoi(argv[i] + 6);
    else if (!strcmp(SubStr(argv[i], 1, 5), "alpha"))
      alpha = atof(argv[i] + 6);
     else if (!strcmp(SubStr(argv[i], 1, 4), "beta"))
      beta = atof(argv[i] + 5);
  }
 
  y_row = x_col;
  z_row = x_row;
  z_col = y_col;
  
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
  
  
  int row, column; // i-row index ,j- column index
  clock_t start, end;
  float *HostMatX; // mxk matrix a on the host
  float *HostMatY; // kxn matrix b on the host
  float *HostMatZ; // mxn matrix c on the host
  
  HostMatX = (float *) malloc (x_row * x_col * sizeof (float)); // host memory for a
  HostMatY = (float *) malloc (y_row * y_col * sizeof (float)); // host memory for b
  HostMatZ = (float *) malloc (z_row * z_col * sizeof (float)); // host memory for c
  
  if (HostMatX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixZ)\n");
    return EXIT_FAILURE;
  }
  
  
  // define an mxk matrix a column by column
  int ind = 11; // a:
  for (column = 0; column < x_col; column++) {                                              // 11 ,17 ,23 ,29 ,35
    for (row = 0; row < x_row; row++) {                                                      // 12 ,18 ,24 ,30 ,36
      HostMatX[index(row, column, x_row)] = (float)ind ++;                                      // 13 ,19 ,25 ,31 ,37
    }                                                                                    // 14 ,20 ,26 ,32 ,38
  }                                                                               // 15 ,21 ,27 ,33 ,39
                                                                                // 16 ,22 ,28 ,34 

  
  
  
  // define a kxn matrix b column by column
  ind = 11; // b:
  for (column = 0; column < y_col; column++) {                                      
    for (row = 0; row < y_row; row++) {                                                
      HostMatY[index(row, column, y_row)] = (float)ind ++;                                           
    }                                                                         
  }                                                       
  
  // define an mxn matrix c column by column
  ind =11; // c:
  for (column = 0; column < z_col; column++) {                             
    for (row = 0; row < z_row; row++) {                                        
      HostMatZ[ index(row, column, z_row )] = (float)ind ++;                  
    }                                                                  
  }
  
  
  std::cout << "\nMatriz X:";
  PrintMat(HostMatX, x_col, x_row);
  std::cout << "\nMatriz Y:\n";
  PrintMat(HostMatY, y_col, y_row);
  std::cout << "\nMatriz Z:\n";
  PrintMat(HostMatZ, z_col, z_row);
  
  
                                                                   
  
  // on the device
  float *DeviceMatX; // d_a - a on the device
  float *DeviceMatY; // d_b - b on the device
  float *DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc ((void **)& DeviceMatX , x_row * x_col * sizeof (*HostMatX)); // device
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory allocation failed for X "<<std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **)& DeviceMatY , y_row * y_col * sizeof (*HostMatY)); // device
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory allocation failed for Y "<<std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **)& DeviceMatZ , z_row * z_col* sizeof (*HostMatZ)); // device
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory allocation failed for Z "<<std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (& handle);           // initialize CUBLAS context
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
  
  // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
  // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
  // al ,bet -scalars
  
  start = clock();
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, x_row, y_col, x_col, &alpha, DeviceMatX,
  x_row, DeviceMatY, y_row, &beta, DeviceMatZ, z_row);
  
  end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  
  
  status = cublasGetMatrix (z_row, z_col, sizeof (*HostMatZ), DeviceMatZ, z_row, HostMatZ, z_row); // cp d_c - >c
   if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix Z from device\n");
    return EXIT_FAILURE;
  }
  
  

  std::cout << "\nMatriz Z after Gemm operation is:\n";
  PrintMat(HostMatZ, z_col, z_row);
  
  
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";
  
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory deallocation failed for X"std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory deallocation failed for X"std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatZ); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout<<" The device memory deallocation failed for X"std::endl;
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy ( handle ); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }
  free (HostMatX); // free host memory
  free (HostMatY); // free host memory
  free (HostMatY); // free host memory
  return EXIT_SUCCESS ;
}
// a:
// 11 17 23 29 35
// 12 18 24 30 36
// 13 19 25 31 37
// 14 20 26 32 38
// 15 21 27 33 39
// 16 22 28 34 40
// b:
// 11 16 21 26
// 12 17 22 27
// 13 18 23 28
// 14 19 24 29
// 15 20 25 30
// c:
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34
// c after Sgemm :
// 1566 2147 2728 3309
// 1632 2238 2844 3450
// 1698 2329 2960 3591 // c=al*a*b+bet *c
// 1764 2420 3076 3732
// 1830 2511 3192 3873
// 1896 2602 3308 4014
