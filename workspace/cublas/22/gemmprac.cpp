#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))   // for getting index values matrices
#define RANDOM (rand() % 1000 * 1.00) / 100    // for getting random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start))


void PrintMatrix(float** Matrix, int batch_count, int matrix_row, int matrix_col) {
  int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[batch][INDEX(row, col, matrix_row) + batch * matrix_row * matrix_col] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {
  clock_t clk_start, clk_end;
  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count;
  float alpha, beta;
    int row,col;
  std::cout << "\n\n" << argv[0] << std::endl;
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
    }

    else if (!(cmd_argument.compare("-A_column"))) {
      A_col = atoi(argv[loop_count + 1]); 
    }

    else if (!(cmd_argument.compare("-B_column"))) {
      B_col = atoi(argv[loop_count + 1]);
    }

    else if (!(cmd_argument.compare("-batch_count"))) {
      batch_count = atoi(argv[loop_count + 1]);
    }
    
    else if (!(cmd_argument.compare("-alpha"))) {
      alpha = atof(argv[loop_count + 1]);
    }

    else if (!(cmd_argument.compare("-beta"))) {
      beta = atof(argv[loop_count + 1]);
    }
  }

  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
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
  float **HostMatA;
  float **HostMatB;
  float **HostMatC;
  
  HostMatA = new float *[batch_count];
  for(int batch = 0; batch < batch_count; batch++) {
    HostMatA[batch] = new float [A_row * A_col];
  }
  
  HostMatB = new float *[batch_count];
  for(int batch = 0; batch < batch_count; batch++) {
    HostMatB[batch] = new float [B_row * B_col];
  }
  
  HostMatC = new float *[batch_count];
  for(int batch = 0; batch < batch_count; batch++) {
    HostMatC[batch] = new float [C_row * C_col];
  }
  
  
  
  
  int ind =11;
  int batch;
  //Setting up values in matrices
  for(batch = 0; batch < batch_count; batch++) {
    for(row = 0; row < A_row; row++) {
      for(col = 0; col < A_col; col++) {
        HostMatA[batch][INDEX(row, col, A_row) + batch * A_col * A_row] = ind++;
      } 
    }
  }
  ind=11;
  for(batch = 0; batch < batch_count; batch++) {
    for(row = 0; row < B_row; row++) {
      for(col = 0; col < B_col; col++) {
        HostMatB[batch][INDEX(row, col, B_row) + batch * B_col * B_row] = ind++;
      } 
    }
  }
  
  ind = 11;
  for(batch = 0; batch < batch_count; batch++) {
    for(row = 0; row < C_row; row++) {
      for(col = 0; col < C_col; col++) {
        HostMatC[batch][INDEX(row, col, C_row) + batch * C_col * C_row] = ind++;
      } 
    }
  }
  
  // Matrix input
  std::cout << "matrix A \n";
  PrintMatrix(HostMatA, batch_count, A_row, A_col); 
  std::cout<<"matrix B\n";
  PrintMatrix(HostMatB, batch_count, B_row, B_col); 
  std::cout<<"matrixC before\n";
  PrintMatrix(HostMatC, batch_count, C_row, C_col);


  // Create host pointer array to device matrix storage
    float **DeviceMatA, **DeviceMatB, **DeviceMatC, **h_d_A, **h_d_B, **h_d_C;
    h_d_A = new float *[batch_count *sizeof(float *)];
    h_d_B = new float *[batch_count *sizeof(float *)];
    h_d_C = new float *[batch_count *sizeof(float *)];
 
    for(row=0; row<batch_count; row++) {
      cudaMalloc((void**)&h_d_A[batch], A_row * A_col * sizeof(float));
      cudaMalloc((void**)&h_d_B[batch], B_row * B_col * sizeof(float));
      cudaMalloc((void**)&h_d_C[batch], C_row * C_col * sizeof(float));
    }
  
   
   
   // Copy the host array of device pointers to the device
    cudaMalloc((void**)&DeviceMatA, batch_count * sizeof(float *));
    cudaMalloc((void**)&DeviceMatB, batch_count * sizeof(float *));
    cudaMalloc((void**)&DeviceMatC, batch_count * sizeof(float *));
    
    cudaMemcpy(DeviceMatA, h_d_A, batch_count*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceMatB, h_d_B, batch_count*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceMatC, h_d_C, batch_count*sizeof(float *), cudaMemcpyHostToDevice);
    
    
    // Set input matrices on device
    for(batch = 0; row<batch_count; row++) {
        cublasSetMatrix(A_row, A_col, sizeof(float), HostMatA[batch], A_row, h_d_A[batch], A_row);
        cublasSetMatrix(B_row, B_col, sizeof(float), HostMatB[batch], B_row, h_d_B[batch], B_row);
        cublasSetMatrix(C_row, C_col, sizeof(float), HostMatC[batch], C_row, h_d_C[batch], C_row);
    }

   // DGEMM: C = alpha*A*B + beta*C
    cublasSgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       A_row, B_col, A_col,
                       &alpha,
                       const float DeviceMatA, A_row,
                       DeviceMatB, B_row,
                       &beta,
                       DeviceMatC, C_row,
                       batch_count);
                       
                       
                       
 
   // Retrieve result matrix from device
    for(batch =0; batch<batch_count; batch++)
        cublasGetMatrix(C_row, C_col, sizeof(float), h_d_C[batch], C_row, HostMatC[batch], C_row);
        
  
   // Matrix output
  std::cout << "\nMatrix C after gemmBatched operation:\n";
  PrintMatrix(HostMatC, batch_count, C_row, C_col);
  
  
  // Clean up resources
  for(batch = 0; batch < batch_count; batch++) {
        free(HostMatA[batch]);
        free(HostMatB[batch]);
        free(HostMatC[batch]);
        cudaFree(h_d_A[batch]);
        cudaFree(h_d_B[batch]);
        cudaFree(h_d_C[batch]);
  }
  
    free(HostMatA);
    free(HostMatB);
    free(HostMatC);
    free(h_d_A);
    free(h_d_B);
    free(h_d_C);
    cudaFree(DeviceMatA);
    cudaFree(DeviceMatB);
    cudaFree(DeviceMatC);
    cublasDestroy(handle);
 
    return 0;
}
    
