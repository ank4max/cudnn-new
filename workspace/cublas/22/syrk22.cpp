#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <string>
#include <time.h>

#define FIRST_ARG "A_row"    //for comparison with command line argument and initializing value of no. of rows for x
#define SECOND_ARG "A_col"   //for comparison with command line argument and initializing value of no. of col for x
#define THIRD_ARG "alpha"   //for comparison with command line argument and initializing value of scalar constant alpha
#define FOURTH_ARG "beta"     //for comparison with command line argument and initializing value of scalar constant beta
#define LEN_ARG_FIRST 5      // defining length for   first cmd line argument for comparison
#define LEN_ARG_SECOND 5     // defining length for  second cmd line argument for comparison
#define LEN_ARG_THIRD 5      // defining length for  third cmd line argument  for comparison
#define LEN_ARG_FOURTH 4     // defining length for  fourth cmd line argument for comparison
#define BEGIN 1              
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(float* Matrix, int matriA_col, int matriA_row) {
  int row, col;
  for (row = 0; row < matriA_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matriA_col; col++) {
      std::cout << Matrix[INDEX(row, col, matriA_row)] << " ";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {
  int A_row, A_col, B_row, B_col;
  float alpha, beta;
  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }
  
  for (int loop_count = 1; loop_count < argc; loop_count++) {
    int len = sizeof(argv[loop_count]);
    std::string str(argv[loop_count]);
    if (!((str.substr(BEGIN, LEN_ARG_FIRST)).compare(FIRST_ARG)))
      A_row = atoi(argv[loop_count] + 6);
    else if (!((str.substr(BEGIN, LEN_ARG_SECOND)).compare(SECOND_ARG)))
      A_col = atoi(argv[loop_count] + LEN_ARG_SECOND + 1);
    else if (!((str.substr(BEGIN, LEN_ARG_THIRD)).compare(THIRD_ARG)))
      alpha = atof(argv[loop_count] + LEN_ARG_THIRD + 1);
    else if (!((str.substr(BEGIN, LEN_ARG_FOURTH)).compare(FOURTH_ARG)))
      beta = atof(argv[loop_count] + LEN_ARG_FOURTH + 1);
  }
  
  B_row = A_row;
  B_col = A_row;

  
  
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ; 
  clock_t clk_start, clk_end;
  int row, col; // i-row index , j- column index
  
  float *HostMatA;                   // nxk matrix a on the host
  float *HostMatB;                   // nxn matrix c on the host
  HostMatA = new float[A_row * A_col]; // host memory for a
  HostMatB = new float[B_row * B_col]; // host memory for c
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an nxn symmetric matrix c
  // column by column
  int ind =11; // c:
  for(col = 0; col < B_col; col++) {                                  // 11
    for(row = 0; row < B_row; row++) {                                // 12 ,17
      if(row >= col) {                                           // 13 ,18 ,22
        HostMatB[INDEX(row, col, B_row)] = (float)ind ++;              // 14 ,19 ,23 ,26
      }                                                     // 15 ,20 ,24 ,27 ,29
    }                                                      // 16 ,21 ,25 ,28 ,30 ,31
  }
  
  // print the lower triangle of c row by row
  std::cout <<" lower triangle of Y:\n";
  for(row = 0; row < B_row; row++) {
    for(col = 0; col < B_col; col++) {
      if(row >= col) {
        std::cout << HostMatB[INDEX(row, col, B_row)] << " ";
      }
    }
    std::cout << "\n";
  }
  
  // define an nxk matrix a column by column
  ind =11; // a:
  for(col = 0; col < A_col; col++) {                          // 11 ,17 ,23 ,29
    for(row = 0; row < A_row; row++) {                        // 12 ,18 ,24 ,30
      HostMatA[INDEX(row, col, A_row)] = (float)ind;        // 13 ,19 ,25 ,31
      ind ++;                                       // 14 ,20 ,26 ,32
    }                                               // 15 ,21 ,27 ,33
  }                                                 // 16 ,22 ,28 ,34

  std::cout << "\nMatriz X:";
  PrintMatrix(HostMatA, A_col, A_row);
  
  // on the device
  float * DeviceMatA; // d_a - a on the device
  float * DeviceMatB; // d_c - c on the device

  
  cudaStatus = cudaMalloc((void **)& DeviceMatA, A_row * A_col * sizeof (*HostMatA)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  // memory alloc for a
  cudaStatus = cudaMalloc((void **)& DeviceMatB, B_row * B_col * sizeof (*HostMatB)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;
  }
  // memory alloc for c
  status = cublasCreate (& handle); // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  // copy matrices from the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA), HostMatA, A_row, DeviceMatA, A_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  } 
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB), HostMatA, B_row, DeviceMatB, B_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  // symmetric rank -k update : c = al*d_a *d_a ^T + bet *d_c ;
  // d_c - symmetric nxn matrix , d_a - general nxk matrix ;
  // al ,bet - scalars
  
  
  clk_start = clock();
  
  status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  A_row, A_col, &alpha, DeviceMatA, A_row, &beta, DeviceMatB, B_row);
  
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  
  status = cublasGetMatrix (B_row, B_col, sizeof (*HostMatB), DeviceMatB, B_row, HostMatB, B_row); // d_c -> c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix Y from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout<<" lower triangle of updated c after Ssyrk :\n";
  for(row = 0; row < B_row; row++) {
    for(col = 0; col < B_col; col++) {
      if(row >= col) {  // print the lower triangle
        std::cout << HostMatB[INDEX(row, col, B_row)] << " " ;  // of c after Ssyrk
      }
    }
    std::cout <<"\n";
  }
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Y" << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasDestroy (handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }
  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  return EXIT_SUCCESS ;
}

// lower triangle of c:
// 11
// 12 17
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
// lower triangle of updated c after Ssyrk : c=al*a*a^T+bet *c
// 1791
// 1872 1961
// 1953 2046 2138
// 2034 2131 2227 2322
// 2115 2216 2316 2415 2513
// 2196 2301 2405 2508 2610 2711

// lower triangle of c:
// 11
// 12 17
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
// lower triangle of updated c after Ssyrk : c=al*a*a^T+bet *c
// 1791
// 1872 1961
// 1953 2046 2138
// 2034 2131 2227 2322
// 2115 2216 2316 2415 2513
// 2196 2301 2405 2508 2610 2711
