#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))   // for getting index values matrices
#define RANDOM (rand() % 1000 * 1.00) / 100    // for getting random values

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
  int A_row, A_col, B_row, B_col, C_row, C_col;
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
  
  //initializing values for matrix B and C
  B_row = A_row;
  B_col = A_col;
  C_row = A_row;
  C_col = A_row;
  
  // creating cublas handle
  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle; 
  clock_t clk_start, clk_end;
  int row, col;
  
  // allocating memory for matrices on host
  float *HostMatA;
  float *HostMatB;
  float *HostMatC;                     
  HostMatA = new float[A_row * A_col]; 
  HostMatB = new float[B_row * B_col];
  HostMatC = new float[C_row * C_col];
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an n x n symmetric matrix C 
  // setting up values for matrix C
  // using RANDOM macro to generate random numbers between 0 - 100
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
  // setting up values for matrix A
  // using RANDOM macro to generate random numbers between 0 - 100
  for(col = 0; col < A_col; col++) {
    for(row = 0; row < A_row; row++) {
      HostMatA[INDEX(row, col, A_row)] = RANDOM;
    }
  }
    
  // define n x k matrix B column by column
  // setting up values for matrix B
  // using RANDOM macro to generate random numbers between 0 - 100
  for(col = 0; col < B_col; col++) {
    for(row = 0; row < B_row; row++) {
      HostMatB[INDEX(row, col, B_row)] = RANDOM;
    }
  }
  
  //printing Matrix A
  std::cout << "\nMatrix A:";
  PrintMatrix(HostMatA, A_row, A_col);
  
  //Printing Matrix B
  std::cout << "\nMatrix B:";
  PrintMatrix(HostMatB, B_row, B_col);
  
  // allocating memory for matrices on device using cudamalloc
  float * DeviceMatA;
  float * DeviceMatB;
  float * DeviceMatC;
  cudaStatus = cudaMalloc((void **)& DeviceMatA, A_row * A_col * sizeof(*HostMatA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc((void **)& DeviceMatB, B_row * B_col * sizeof(*HostMatB));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B\n";
    return EXIT_FAILURE;
  }
  
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

  // setting the values of matrices on device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA), HostMatA, A_row, DeviceMatA, A_row); // A -> d_A
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  }
 
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB), HostMatB, B_row, DeviceMatB, B_row); // A -> d_A
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  // start variable to store time
  clk_start = clock();
  
  status = cublasSsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  A_row, A_col, &alpha, DeviceMatA, A_row, DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  // end variable to store time
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << status << std::endl;
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  // getting the final output
  status = cublasGetMatrix(C_row, C_col, sizeof (*HostMatC), DeviceMatC, C_row, HostMatC, C_row); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  // Matrix output
  std::cout<<"\nLower triangle of updated C after syrkx :\n";
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
  
  //free device memory
  cudaStatus = cudaFree (DeviceMatA); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatB); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatC); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C\n";
    return EXIT_FAILURE;   
  }
  
  // destroying cublas handle
  status  = cublasDestroy (handle); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  } 
  
  // freeing host memory
  delete[] HostMatA; 
  delete[] HostMatB; 
  delete[] HostMatC; 
  return EXIT_SUCCESS;
}
  
  
