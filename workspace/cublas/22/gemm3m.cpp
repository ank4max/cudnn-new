#include <iostream>
#include <string>
#include "cublas.h"
#include "cublas_v2.h"
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values 

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(cuComplex* Matrix, int matriA_row, int matriA_col) {
  int row, col;
  for (row = 0; row < matriA_row; row++) {
    for (col = 0; col < matriA_col; col++) {
      std::cout << Matrix[INDEX(row, col, matriA_row)].x << "+" << Matrix[INDEX(row, col, matriA_row)].y << "*I ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha_real, alpha_imaginary, beta_real, beta_imaginary;

  std::cout << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // reading cmd line arguments
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = atof(argv[loop_count + 1]);
  }
  
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  alpha_imaginary = 0.0f;
  beta_imaginary = 0.0f;
  
  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle;
  clock_t clk_start, clk_end;
  int row, col;
  
  cuComplex *HostMatA; // mxk complex matrix a on the host
  cuComplex *HostMatB; // kxn complex matrix b on the host
  cuComplex *HostMatC; // mxn complex matrix c on the host
  HostMatA = new cuComplex[A_row * A_col]; // host memory
  // alloc for A
  HostMatB = new cuComplex[B_row * B_col]; // host memory
  // alloc for B
  HostMatC = new cuComplex[C_row * C_col]; // host memory
  // alloc for C
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix A)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix B)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix C)\n");
    return EXIT_FAILURE;
  }

  //defining a matrix A
  int ind =11;
  for (col = 0; col < A_col; col++) {                 
    for (row = 0; row < A_row; row++) {                                                                           
      HostMatA[INDEX(row, col, A_row)].x = ( float )ind ++;                    
      HostMatA[INDEX(row, col, A_row)].y = 0.0f;                       
                                                               
    }
  }
  
  //printing Matrix A 
  std::cout << "Matrix  A :\n";
  PrintMatrix(HostMatA, A_row, A_col);
  
  // define kxn matrices b column by column
  // b:
  ind =11;
  for(col = 0; col < B_col; col++) {           
    for(row = 0; row < B_row; row++) {                      
      HostMatB[INDEX(row, col, B_row)].x = ( float )ind ++;            
      HostMatB[INDEX(row, col, B_row)].y = 0.0f;                   
                   
    }
  }
  
  // define mxn matrices c column by column 
  ind =11; 
  for(col = 0; col < C_col; col++) {           
    for(row = 0; row < C_row; row++) {                      
      HostMatC[INDEX(row, col, C_row)].x = ( float )ind ++;              
      HostMatC[INDEX(row, col, C_row)].y = 0.0f;                 
    }
  }
  
  
  //printing mat B and mat C
  // print B row by row
  std::cout << "B:\n";
  PrintMatrix(HostMatC, C_row, C_col);
  // print c row by row
  std::cout << "C:\n";
  PrintMatrix(HostMatC, C_row, C_col);
  
  // on the device
  cuComplex * DeviceMatA; // d_a - a on the device
  cuComplex * DeviceMatB; // d_b - b on the device
  cuComplex * DeviceMatC; // d_c - c on the device
  cudaStatus = cudaMalloc ((void **)& DeviceMatA , A_row * A_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  // device memory alloc for a
  cudaStatus = cudaMalloc ((void **)& DeviceMatB , B_row * B_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for b
  cudaStatus = cudaMalloc ((void **)& DeviceMatC, C_row * C_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for c
  
  status = cublasCreate (& handle);  // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  // copy matrices from the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA) , HostMatA, A_row, DeviceMatA, A_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB) , HostMatB, B_row, DeviceMatB, B_row); //b -> d_b
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC) , HostMatC, C_row, DeviceMatC, C_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }
           
  cuComplex alpha ={alpha_real, alpha_imaginary}; // al =1
  cuComplex beta ={beta_real, beta_imaginary}; // bet =1
  clk_start = clock();

  // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
  // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
  // alpha, beta - scalars
  
  status = cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, &alpha, DeviceMatA, A_row,
                        DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC),
                            DeviceMatC, C_row, HostMatC, C_row); // copy d_z -> C
           
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after Gemm3mm operation is:\n";
  PrintMatrix(HostMatC, C_row, C_col);
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " <<THROUGHPUT(clk_start, clk_end) << "\n\n";
  cudaStatus = cudaFree (DeviceMatA); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatB); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatC); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy (handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }

  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  delete[] HostMatC; // free host memory

  return EXIT_SUCCESS ;
}
  
  



