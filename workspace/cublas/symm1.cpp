#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>
#define IDX2C(i ,j , ld ) ((( j )*( ld ))+( i ))
#define m 6 // a - mxm matrix
#define n 4 // b,c - mxn matrices

char* SubStr(char* InputArr, int begin, int len) {
  char* ResultStr = new char[len + 1];
  for (int i = 0; i < len; i++) {
    ResultStr[i] = *(InputArr + begin + i);
  }
  ResultStr[len] = 0;
  return ResultStr;
}

int main (int argc, char **argv) {
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
  clock_t start, end;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  
  int i,j; // i-row ind. , j- column ind.
  float * HostMatX; // mxm matrix a on the host
  float * HostMatY; // mxn matrix b on the host
  float * HostMatZ; // mxn matrix c on the host
  
  HostMatX = (float *) malloc (m*m* sizeof (float)); // host memory for a
  HostMatY = (float *) malloc (m*n* sizeof (float)); // host memory for b
  HostMatZ = (float *) malloc (m*n* sizeof (float)); // host memory for c
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
  for(j = 0; j < m; j++) {                     // 11
    for(i = 0; i < m; i++) {                   // 12 ,17
      if(i >=j) {                              // 13 ,18 ,22
        HostMatX[IDX2C (i, j, m)] = (float)ind ++;    // 14 ,19 ,23 ,26
      }                                        // 15 ,20 ,24 ,27 ,29
    }                                          // 16 ,21 ,25 ,28 ,30 ,31
  }
  // print the lower triangle of a row by row
  printf (" lower triangle of x:\n");
  
  for(i = 0; i < m; i++) {
    for(j = 0; j < m; j++) {
      if(i >=j) {
        printf (" %5.0f", HostMatX[IDX2C (i, j, m)]);
      }
      
    }
    printf ("\n");
  }
  // define mxn matrices b,c column by column
  ind =11; // b,c:
  for(j = 0; j < n; j++) {                // 11 ,17 ,23 ,29
    for(i = 0; i < m; i++) {                        // 12 ,18 ,24 ,30
      HostMatY[IDX2C (i, j, m)]=( float )ind;         // 13 ,19 ,25 ,31
      HostMatZ[IDX2C (i, j, m)]=( float )ind;         // 14 ,20 ,26 ,32
      ind ++; // 15 ,21 ,27 ,33
    } // 16 ,22 ,28 ,34
  }
  // print b(=c) row by row
  printf ("b(=c):\n");
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j ++) {
      printf (" %5.0f", HostMatY[IDX2C (i, j, m)]);
    }
    printf ("\n");
  }
  // on the device
  float *DeviceMatX; // d_a - a on the device
  float *DeviceMatY; // d_b - b on the device
  float *DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc((void **)& DeviceMatX, m*m* sizeof (*HostMatX)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for X\n");
    return EXIT_FAILURE;
  }
  // memory alloc for a
  cudaStatus = cudaMalloc((void **)& DeviceMatY, m*n* sizeof (*HostMatY)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Y\n");
    return EXIT_FAILURE;
  }
  // memory alloc for b
  cudaStatus = cudaMalloc((void **)& DeviceMatZ, m*n* sizeof (*HostMatZ)); // device
  if(cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Z\n");
    return EXIT_FAILURE;
  }
  // memory alloc for c
  status = cublasCreate (& handle); // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  // copy matrices from the host to the device
  status = cublasSetMatrix (m, m, sizeof (*HostMatX), HostMatX, m, DeviceMatX, m); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (m, n, sizeof (*HostMatY), HostMatY, m, DeviceMatY, m); //b -> d_b
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed\n");
    return EXIT_FAILURE;
  }
 
  status = cublasSetMatrix (m, n, sizeof (*HostMatZ), HostMatZ, m, DeviceMatZ, m); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed\n");
    return EXIT_FAILURE;
  }
  float al =1.0f; // al =1
  float bet =1.0f; // bet =1
  
  
  // symmetric matrix - matrix multiplication :
  // d_c = al*d_a *d_b + bet *d_c ; d_a - mxm symmetric matrix ;
  // d_b ,d_c - mxn general matrices ; al ,bet - scalars;
  
  start = clock();
  
  status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
  m, n, &al, DeviceMatX, m, DeviceMatY, m, &bet, DeviceMatZ, m);
  
  end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  status = cublasGetMatrix (m,n, sizeof (*HostMatZ), DeviceMatZ, m, HostMatZ, m); // d_c -> c
  printf ("c after Ssymm :\n"); // print c after Ssymm
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j ++) {
      printf (" %7.0f",HostMatZ[ IDX2C (i,j,m )]);
    }
    printf ("\n");
  }
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";
  
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed for X\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed for Y\n");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatZ); // free device memory
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed for Z\n");
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy ( handle ); // destroy CUBLAS context
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
