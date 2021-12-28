#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>
#define IDX2C(i,j,ld) ((( j )*( ld ))+( i ))

#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix

int main (int argc, char **argv  ) {
  
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
  
  
  int row,column; // i-row index ,j- column index
  clock_t start, end;
  float *HostMatX; // mxk matrix a on the host
  float *HostMatY; // kxn matrix b on the host
  float *HostMatZ; // mxn matrix c on the host
  
  HostMatX=( float *) malloc (m*k* sizeof ( float )); // host memory for a
  HostMatY=( float *) malloc (k*n* sizeof ( float )); // host memory for b
  HostMatZ=( float *) malloc (m*n* sizeof ( float )); // host memory for c
  
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
  
  
  // define an mxk matrix a column by column
  int ind =11; // a:
  for(column = 0; column < k; column++) {                                              // 11 ,17 ,23 ,29 ,35
    for(row = 0; row < m; row++) {                                                      // 12 ,18 ,24 ,30 ,36
      HostMatX[ IDX2C (row,column,m )]=( float )ind ++;                                      // 13 ,19 ,25 ,31 ,37
    }                                                                                    // 14 ,20 ,26 ,32 ,38
  }                                                                               // 15 ,21 ,27 ,33 ,39
                                                                                // 16 ,22 ,28 ,34 ,40
  
  
  
  
  // print a row by row
  printf ("X:\n");
  for (row = 0; row < m; row ++) {
    for (column = 0; column < k; column++) {
      printf (" %5.0f",HostMatX[ IDX2C (row,column,m )]);
    }
    printf ("\n");
  }
  // define a kxn matrix b column by column
  ind =11; // b:
  for(column = 0; column < n; column++) {                                      // 11 ,16 ,21 ,26
    for(row = 0; row < k; row++) {                                                // 12 ,17 ,22 ,27
      HostMatY[ IDX2C (row,column,k )]=( float )ind ++;                                           // 13 ,18 ,23 ,28
    }                                                                         // 14 ,19 ,24 ,29
  }                                                       // 15 ,20 ,25 ,30
  // print b row by row
  printf ("Y:\n");
  for (row = 0; row < k; row++) {
    for (column = 0; column < n; column++) {
      printf (" %5.0f",HostMatY[ IDX2C (row,column,k )]);
    }
    printf ("\n");
  }
  
  // define an mxn matrix c column by column
  ind =11; // c:
  for (column = 0; column < n; column++) {                             // 11 ,17 ,23 ,29
    for (row = 0; row < m; row++) {                                        // 12 ,18 ,24 ,30
      HostMatZ[ IDX2C (row,column,m )]=( float )ind ++;                  // 13 ,19 ,25 ,31
    }                                                                  // 14 ,20 ,26 ,32
  }                                                                     // 15 ,21 ,27 ,33
                                                                   // 16 ,22 ,28 ,34
  // print c row by row
  printf ("Z:\n");
  for (row = 0; row < m; row++) {
    for (column = 0; column < n; column++) {
      printf (" %5.0f",HostMatZ[ IDX2C (row,column,m )]);
    }
    printf ("\n");
  }
  // on the device
  float *DeviceMatX; // d_a - a on the device
  float *DeviceMatY; // d_b - b on the device
  float *DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc (( void **)& DeviceMatX , m*k* sizeof (*HostMatX)); // device
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for X \n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc (( void **)& DeviceMatY , k*n* sizeof (*HostMatY)); // device
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Y\n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc (( void **)& DeviceMatZ , m*n* sizeof (*HostMatZ)); // device
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed for Z\n");
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (& handle );           // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  // copy matrices from the host to the device
  status = cublasSetMatrix (m, k, sizeof (*HostMatX), HostMatX, m, DeviceMatX, m); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetMatrix (k, n, sizeof (*HostMatY), HostMatY, k, DeviceMatY, k); //b -> d_b
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
  // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
  // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
  // al ,bet -scalars
  
  start = clock();
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, DeviceMatX,
  m, DeviceMatY, k, &bet, DeviceMatZ, m);
  
  end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  
  
  status = cublasGetMatrix (m, n, sizeof (*HostMatZ), DeviceMatZ, m, HostMatZ, m); // cp d_c - >c
   if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix Z from device\n");
    return EXIT_FAILURE;
  }
  printf ("c after Sgemm :\n");
  for(row = 0; row < m; row ++) {
    for(column = 0; column < n; column ++) {
      printf (" %7.0f",HostMatZ[ IDX2C (row,column,m )]); // print c after Sgemm
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
