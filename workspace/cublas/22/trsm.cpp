#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))     // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values 

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(float* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {
  clock_t clk_start, clk_end;
  int A_row, A_col, B_row, B_col;
  float alpha;

  std::cout << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // reading command line arguments
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);

    if (!(cmd_argument.compare("-A_row"))) {
      A_row = atoi(argv[loop_count + 1]); 
      B_row = A_row;
      A_col = A_row;
    }
    else if (!(cmd_argument.compare("-B_column"))) {
      B_col = atoi(argv[loop_count + 1]);
    }
    else if (!(cmd_argument.compare("-alpha"))) {
      alpha = atoi(argv[loop_count + 1]);
    }
  }
  
  // creating cublas handle
  cudaError_t cudaStatus;
  cublasStatus_t status;
  cublasHandle_t handle;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // allocating memory for matrices on host
  float *HostMatA = new float[A_row * A_col];
  float *HostMatB = new float[B_row * B_col];

  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  // using RANDOM macro to generate random numbers between 0 - 100
  int row, col;
  for (col = 0; col < A_col; col++) {
    for (row = 0; row < A_row; row++) {
      if (row >= col) 
        HostMatA[INDEX(row, col, A_row)] = RANDOM;
      else 
        HostMatA[INDEX(row, col, A_row)] = 0.0;
    }
  }

  for (row = 0; row < B_row; row++) {
    for (col = 0; col < B_col; col++) {
        HostMatB[INDEX(row, col, B_row)] = RANDOM;
    }
  }

  std::cout << "\nMatrix A:";
  PrintMatrix(HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:";
  PrintMatrix(HostMatB, B_row, B_col);

  // allocating memory for matrices on device using cublasAlloc
  float* DeviceMatA;
  float* DeviceMatB;

  status = cublasAlloc(A_row * A_col, sizeof(float), (void**) &DeviceMatA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(B_row * B_col, sizeof(float), (void**) &DeviceMatB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(A_row, A_col, sizeof(float), HostMatA, A_row, DeviceMatA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(B_row, B_col, sizeof(float), HostMatB, B_row, DeviceMatB, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (B) failed\n");
    return EXIT_FAILURE;
  }
  
  // start variable to store time
  clk_start = clock();
  
  // solve d_A * X = alpha * d_B
  // the solution X overwrites rhs d_B
  // d_A - m x m triangular matrix in lower mode
  // d_B, X - m x n general matrices
  // alpha - scalar
  status = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, &alpha, 
                       DeviceMatA, A_row, DeviceMatB, B_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! API execution failed\n");
    return EXIT_FAILURE;
  }

  // end variable to store time
  clk_end = clock();

  // getting the final output
  status = cublasGetMatrix(B_row, B_col, sizeof(float), DeviceMatB, B_row, HostMatB, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host Matrix B\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  std::cout << "\nMatrix B after trsm operation:";
  PrintMatrix(HostMatB, B_row, B_col);  

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  // free device memory
  cudaStatus = cudaFree(DeviceMatA);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatB);
  if( cudaStatus != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }

  // destroying cublas handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize");
    return EXIT_FAILURE;
  }

  // freeing host memory
  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  
  return EXIT_SUCCESS ;
}
