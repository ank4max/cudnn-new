#include <iostream>
#include <string>
#include "cublas_v2.h"
#include<cuda_runtime.h>
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 
cudaError_t cudaStatus; 
cublasStatus_t status; 
cublasHandle_t handle;
clock_t clk_start, clk_end;

template< class Data >
void PrintMatrix(Data* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
    }
  }
  std::cout << "\n";
}

void PrintCuMatrix(cuComplex* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
    }
    std::cout << "\n";
  }
}




int sgemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, float alpha, float beta) {
  
  clk_start = 0;
  clk_end = 0;
 
  float *HostMatA; // mxk matrix A on the host
  float *HostMatB; // kxn matrix B on the host
  float *HostMatC; // mxn matrix C on the host
  
  HostMatA = new float[A_row * A_col]; // host memory for A
  HostMatB = new float[B_row * B_col]; // host memory for B
  HostMatC = new float[C_row * C_col]; // host memory for C
  
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

  int row, col;
  
  // define an mxk matrix A column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < A_row; row++) {                                              
    for (col = 0; col < A_col; col++) {                                                   
      HostMatA[INDEX(row, col, A_row)] = RANDOM;                                      
    }                                                                                    
  }                                                                               
                                                                               
  // define a kxn matrix B column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < B_row; row++) {                                      
    for (col = 0; col < B_col; col++) {                                                
      HostMatB[INDEX(row, col, B_row)] = RANDOM;                                           
    }                                                                         
  }                                                       
  
  // define an mxn matrix C column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < C_row; row++) {                             
    for (col = 0; col < C_col; col++) {                                        
      HostMatC[INDEX(row, col, C_row)] = RANDOM;                 
    }                                                                  
  }
  
  // printing input matrices
  std::cout << "\nMatrix A:\n";
  PrintMatrix <float> (HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:\n";
  PrintMatrix <float> (HostMatB, B_row, B_col);
  std::cout << "\nMatrix C:\n";
  PrintMatrix <float> (HostMatC, C_row, C_col);

  // on the device
  float *DeviceMatA; // d_A - A on the device
  float *DeviceMatB; // d_B - B on the device
  float *DeviceMatC; // d_C - C on the device

  cudaStatus = cudaMalloc ((void **) &DeviceMatA , A_row * A_col * sizeof (*HostMatA));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatB , B_row * B_col * sizeof (*HostMatB));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatC , C_row * C_col * sizeof (*HostMatC));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (&handle);      // initialize CUBLAS context
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
  
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB), HostMatB, B_row, DeviceMatB, B_row); // B -> d_B
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    return EXIT_FAILURE;
  }
  

  clk_start = clock();

  // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
  // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
  // alpha, beta - scalars
  status = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, &alpha, DeviceMatA, A_row,
                        DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();
  
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC),
                            DeviceMatC, C_row, HostMatC, C_row); // copy d_z -> C

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after sGemm operation is:\n";
  PrintMatrix <float> (HostMatC, C_row, C_col); 
  
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
  return 0;
}
  
int dgemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha, double beta) {
  
  clk_start = 0;
  clk_end = 0;
 
  double*HostMatA; // mxk matrix A on the host
  double*HostMatB; // kxn matrix B on the host
  double*HostMatC; // mxn matrix C on the host
  
  HostMatA = new double[A_row * A_col]; // host memory for A
  HostMatB = new double[B_row * B_col]; // host memory for B
  HostMatC = new double[C_row * C_col]; // host memory for C
  
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

  int row, col;
  
  // define an mxk matrix A column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < A_row; row++) {                                              
    for (col = 0; col < A_col; col++) {                                                   
      HostMatA[INDEX(row, col, A_row)] = RANDOM;                                      
    }                                                                                    
  }                                                                               
                                                                               
  // define a kxn matrix B column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < B_row; row++) {                                      
    for (col = 0; col < B_col; col++) {                                                
      HostMatB[INDEX(row, col, B_row)] = RANDOM;                                           
    }                                                                         
  }                                                       
  
  // define an mxn matrix C column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < C_row; row++) {                             
    for (col = 0; col < C_col; col++) {                                        
      HostMatC[INDEX(row, col, C_row)] = RANDOM;                 
    }                                                                  
  }
  
  // printing input matrices
  std::cout << "\nMatrix A:\n";
  PrintMatrix <double> (HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:\n";
  PrintMatrix <double> (HostMatB, B_row, B_col);
  std::cout << "\nMatrix C:\n";
  PrintMatrix <double> (HostMatC, C_row, C_col);

  // on the device
  double*DeviceMatA; // d_A - A on the device
  double*DeviceMatB; // d_B - B on the device
  double*DeviceMatC; // d_C - C on the device

  cudaStatus = cudaMalloc ((void **) &DeviceMatA , A_row * A_col * sizeof (*HostMatA));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatB , B_row * B_col * sizeof (*HostMatB));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatC , C_row * C_col * sizeof (*HostMatC));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (&handle);      // initialize CUBLAS context
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
  
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB), HostMatB, B_row, DeviceMatB, B_row); // B -> d_B
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    return EXIT_FAILURE;
  }
  

  clk_start = clock();

  // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
  // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
  // alpha, beta - scalars
  status = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, &alpha, DeviceMatA, A_row,
                        DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! DGemm kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();
  
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC),
                            DeviceMatC, C_row, HostMatC, C_row); // copy d_z -> C

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after Gemm operation is:\n";
  PrintMatrix <double> (HostMatC, C_row, C_col); 
  
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
  return 0;
}

int Cgemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, float alpha_real, float beta_real) {
  clk_start = 0;
  clk_end = 0;
  float alpha_imaginary = 0.0f;
  float beta_imaginary = 0.0f; 
  cuComplex alpha = {alpha_real,alpha_imaginary};
  cuComplex beta = {beta_real, beta_imaginary};
 
  cuComplex*HostMatA; // mxk matrix A on the host
  cuComplex*HostMatB; // kxn matrix B on the host
  cuComplex*HostMatC; // mxn matrix C on the host
  
  HostMatA = new cuComplex[A_row * A_col]; // host memory for A
  HostMatB = new cuComplex[B_row * B_col]; // host memory for B
  HostMatC = new cuComplex[C_row * C_col]; // host memory for C
  
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

  int row, col;
  
  // define an mxk matrix A column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < A_row; row++) {                                              
    for (col = 0; col < A_col; col++) {                                                   
      HostMatA[INDEX(row, col, A_row)].x = RANDOM; 
      HostMatA[INDEX(row, col, A_row)].y = 0.0f;
    }                                                                                    
  }                                                                               
                                                                               
  // define a kxn matrix B column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < B_row; row++) {                                      
    for (col = 0; col < B_col; col++) {                                                
      HostMatB[INDEX(row, col, B_row)].x = RANDOM; 
      HostMatB[INDEX(row, col, B_row)].y = 0.0f;
    }                                                                         
  }                                                       
  
  // define an mxn matrix C column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < C_row; row++) {                             
    for (col = 0; col < C_col; col++) {                                        
      HostMatC[INDEX(row, col, C_row)].x = RANDOM;
      HostMatC[INDEX(row, col, C_row)].y = 0.0f;
    }                                                                  
  }
  
  // printing input matrices
  std::cout << "\nMatrix A:\n";
  PrintCuMatrix(HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:\n";
  PrintCuMatrix(HostMatB, B_row, B_col);
  std::cout << "\nMatrix C:\n";
  PrintCuMatrix(HostMatC, C_row, C_col);

  // on the device
  cuComplex*DeviceMatA; // d_A - A on the device
  cuComplex*DeviceMatB; // d_B - B on the device
  cuComplex*DeviceMatC; // d_C - C on the device

  cudaStatus = cudaMalloc ((void **) &DeviceMatA , A_row * A_col * sizeof (*HostMatA));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatB , B_row * B_col * sizeof (*HostMatB));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatC , C_row * C_col * sizeof (*HostMatC));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (&handle);      // initialize CUBLAS context
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
  
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB), HostMatB, B_row, DeviceMatB, B_row); // B -> d_B
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    return EXIT_FAILURE;
  }
  

  clk_start = clock();

  // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
  // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
  // alpha, beta - scalars
  status = cublasCgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, &alpha, DeviceMatA, A_row,
                        DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! CGemm kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();
  
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC),
                            DeviceMatC, C_row, HostMatC, C_row); // copy d_z -> C

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after CGemm operation is:\n";
  PrintCuMatrix(HostMatC, C_row, C_col); 
  
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
  return 0;
}
           




int main (int argc, char **argv) {
  
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha, beta;
  char n;

  std::cout << "\n\n" << argv[0] << std::endl;
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

    else if (!(cmd_argument.compare("-alpha")))
      alpha = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = atof(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-mode")))
      n = *(argv[loop_count + 1]);
  }
 
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  //function call
  
  switch (n) {
  case 's' :
    std::cout << "Calling sGemm function\n";
    sgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta);
    break;
  
  case 'd' :
    std::cout << "Calling DGemm function\n";
    dgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta);
    break;
                        
 case 'c' :
    std::cout << "Calling CGemm function\n";
    Cgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta);
    break; 
  
  }
  return EXIT_SUCCESS;
}
