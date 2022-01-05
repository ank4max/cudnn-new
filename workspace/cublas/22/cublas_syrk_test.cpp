

#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))   // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // for getting random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(float* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << std::endl;
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
    }
  }
  std::cout << std::endl;
}

int main (int argc, char **argv) {
  int A_row, A_col, C_row, C_col;
  float alpha, beta;
  
  std::cout << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // for reading command line arguements
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha")))
      alpha = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = atof(argv[loop_count + 1]);
  }
  
  C_row = A_row;
  C_col = A_row;

  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle; 
  clock_t clk_start, clk_end;
  
  float *HostMatA;                     // nxk matrix A on the host
  float *HostMatC;                     // nxn matrix C on the host
  HostMatA = new float[A_row * A_col]; // host memory for A
  HostMatC = new float[C_row * C_col]; // host memory for C
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an n x n symmetric matrix C
  // column by column
  int row, col;
  for(col = 0; col < C_col; col++) {
    for(row = 0; row < C_row; row++) {
      if(row >= col) {
        HostMatC[INDEX(row, col, C_row)] = RANDOM;
      }
    }
  }
  
  // print the lower triangle of C row by row
  std::cout <<"\nLower triangle of C:\n";
  for(row = 0; row < C_row; row++) {
    for(col = 0; col < C_col; col++) {
      if(row >= col) {
        std::cout << HostMatC[INDEX(row, col, C_row)] << " ";
      }
    }
    std::cout << "\n";
  }
  
  // define n x k matrix A column by column
  for(col = 0; col < A_col; col++) {
    for(row = 0; row < A_row; row++) {
      HostMatA[INDEX(row, col, A_row)] = RANDOM;
    }
  }

  std::cout << "\nMatrix A:";
  PrintMatrix(HostMatA, A_row, A_col);
  
  float * DeviceMatA;
  float * DeviceMatC;

  // memory alloc for A
  cudaStatus = cudaMalloc((void **)& DeviceMatA, A_row * A_col * sizeof(*HostMatA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }

  // memory alloc for C
  cudaStatus = cudaMalloc((void **)& DeviceMatC, C_row * C_col * sizeof(*HostMatC));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C\n";
    return EXIT_FAILURE;
  }
  // initialize CUBLAS context
  status = cublasCreate (&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // copy matrices from the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA), HostMatA, A_row, DeviceMatA, A_row); // A -> d_A
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  } 

  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }

  clk_start = clock();

  // symmetric rank-k update : C = alpha * d_A * d_A^T + beta * d_C ;
  // d_C - symmetric n x n matrix, d_A - general n x k matrix ;
  // alpha, beta - scalars  
  status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                       A_row, A_col, &alpha, DeviceMatA, A_row, &beta, DeviceMatC, C_row);
  
  clk_end = clock();

  if (status != CUBLAS_STATUS_SUCCESS) {
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
