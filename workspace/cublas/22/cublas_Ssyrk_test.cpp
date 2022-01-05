// nvcc 038 ssyrk .c -lcublas
# include <stdio.h>
#include<iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# define INDEX(row, col, row_count) ((( col )*(row_count ))+( row ))
#define RANDOM (rand() % 10000 * 1.00) / 100    // for getting random values
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start))
# define n 6 // a - nxk matrix
# define k 4 // c - nxn matrix


int main ( void ){
  
  int A_row, A_col, C_row, C_col;
  float alpha , beta;
  
  A_row = n;
  A_col = k;
  C_row = A_row;
  C_col = C_row;
  alpha = 1.0f;
  beta  = 1.0f;
  clock_t clk_start, clk_end;
  
  
  
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLASfunctions status
  cublasHandle_t handle ; // CUBLAS context
  int row, col; // i-row index , j- column index
  float *HostMatA;         // nxk matrix a on the host
  float *HostMatC;        // nxn matrix c on the host
  HostMatA = new float[A_row * A_col]; // host memoryfor a
  HostMatC = new float[C_row * C_col]; // host memoryfor c
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }
  
  
  // define the lower triangle of an nxn symmetric matrix c
  // column by column
  int ind =11; // c:
  for(col = 0; col < C_col; col++) { // 11
    for(row = 0; row < C_row; row++) { // 12 ,17
      if(row >= col) { // 13 ,18 ,22
        HostMatC[INDEX(row, col, C_row)] = (float )ind ++; // 14 ,19 ,23 ,26
      } // 15 ,20 ,24 ,27 ,29
    } // 16 ,21 ,25 ,28 ,30 ,31
  }
  // print the lower triangle of c row by row
  std::cout << " lower triangle of c:\n";
  for (row = 0; row < C_row; row++) {
    for (col = 0; col < C_col; col++) {
      if(row >= col)
        std::cout << HostMatC[INDEX(row, col, C_row )];
    }
    std::cout<<"\n";
  }
  // define an nxk matrix a column by column
  ind =11; // a:
  for(col = 0; col < A_col; col++) { // 11 ,17 ,23 ,29
    for(row = 0; row < A_row; row++) { // 12 ,18 ,24 ,30
      HostMatA[INDEX(row, col, A_row)] = (float )ind; // 13 ,19 ,25 ,31
     ind ++; // 14 ,20 ,26 ,32
    } // 15 ,21 ,27 ,33
  } // 16 ,22 ,28 ,34
  std::cout <<"a:\n";
  for (row = 0; row < A_row; row++) {
    for (col = 0; col < A_col; col++) {
      std::cout << HostMatA[INDEX(row, col, A_row )]; // print a row by row
    }
    std::cout << "\n";
  }

  // on the device
  float * DeviceMatA; // d_a - a on the device
  float * DeviceMatC;

  cudaStatus = cudaMalloc (( void **)& DeviceMatA ,A_row * A_col * sizeof (*HostMatA)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  // memory allocfor a
  cudaStatus = cudaMalloc (( void **)& DeviceMatC ,C_row * C_col * sizeof (*HostMatC)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C\n";
    return EXIT_FAILURE;
  }
  // memory allocfor c
  status = cublasCreate (& handle ); // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  // copy matricesfrom the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA) ,HostMatA, A_row, DeviceMatA, A_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  } 
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC ,C_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  } 
  
  
  // symmetric rank -k update : c = al*d_a *d_a ^T + bet *d_c ;
  // d_c - symmetric nxn matrix , d_a - general nxk matrix ;
  // al ,bet - scalars
  clk_start = clock();
  status = cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  A_row, A_col, &alpha, DeviceMatA, A_row, &beta, DeviceMatC, C_row);
  
  clk_end = clock();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << status << std::endl;
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  

  status = cublasGetMatrix(C_row, C_col, sizeof (*HostMatC), DeviceMatC, C_row, HostMatC, C_row); // d_C -> C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout<<"\nLower triangle of updated C after syrk :\n";
  for(row = 0; row < C_row; row++) {
    for(col = 0; col < C_col; col++) {
      if(row >= col) {
        std::cout << HostMatC[INDEX(row, col, C_row)] << " " ;
      }
    }
    std::cout <<"\n";
  }
  
  // Printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  
  cudaStatus = cudaFree(DeviceMatA); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatC); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasDestroy(handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }
  
  delete[] HostMatA; // free host memory
  delete[] HostMatC; // free host memory

  return EXIT_SUCCESS ;
}

// lower triangle of c:
// 11
// 12 17
//n
// 13 18 22
// 14 19 23 26
// 15 20 24 27 29
// 16 21 25 28 30 31

// a:
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34

// lower triangle of updated c after Ssyrk : c = al * a * a^T + bet * c
// 1791
// 1872 1961
// 1953 2046 2138
// 2034 2131 2227 2322
// 2115 2216 2316 2415 2513
// 2196 2301 2405 2508 2610 2711
