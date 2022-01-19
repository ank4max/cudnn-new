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

 template< class D >
 void DefineMat(D* Matrix, int matrix_row, int matrix_col) {
   int row , col;  
   for (row = 0; row < matrix_row; row++) {                                              
    for (col = 0; col < matrix_col; col++) {                                                   
      Matrix[INDEX(row, col, matrix_row)] = RANDOM;                                      
    }                                                                                    
  }                                                                               
 }

 template <class C >
 void DefineCuMat(C* Matrix, int matrix_row, int matrix_col) {
   int row, col;  
   for(col = 0; col < matrix_col; col++) {           
    for(row = 0; row < matrix_row; row++) {                      
      Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
      Matrix[INDEX(row, col, matrix_row)].y = 0.0f;             
      
    }
  }
 }


template <class P>
void PrintCuMatrix(P* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I "    ;
    }
    std::cout << "\n";
  } 
}

template < class T, class T1>
int Gemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T1 alpha_real, T1 alpha_imaginary, T1 beta_real, T1 beta_imaginary, char n) {
  
  clk_start = 0;
  clk_end = 0;
  cuComplex alpha;
  cuComplex beta;
  cuDoubleComplex alpha_z ; 
  cuDoubleComplex beta_z;

  
  T *HostMatA; // mxk matrix A on the host
  T *HostMatB; // kxn matrix B on the host
  T *HostMatC; // mxn matrix C on the host
  
  HostMatA = new T[A_row * A_col]; // host memory for A
  HostMatB = new T[B_row * B_col]; // host memory for B
  HostMatC = new T[C_row * C_col]; // host memory for C
  
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
  
  // define an mxk matrix A,B,C column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  if(n == 's') {
    DefineMat<float>((float *)HostMatA, A_row, A_col);
    DefineMat<float>((float *)HostMatB, B_row, B_col);
    DefineMat<float>((float *)HostMatC, C_row, C_col);

    // printing input matrices
    std::cout << "\nMatrix A:\n";
    PrintMatrix <float> ((float *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintMatrix <float> ((float *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintMatrix <float> ((float *)HostMatC, C_row, C_col);

  }

  else if (n == 'd') {
    DefineMat<double>((double *)HostMatA, A_row, A_col);
    DefineMat<double>((double *)HostMatB, B_row, B_col);
    DefineMat<double>((double *)HostMatC, C_row, C_col);

    // printing input matrices
    std::cout << "\nMatrix A:\n";
    PrintMatrix <double> ((double *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintMatrix <double> ((double *)HostMatB, B_row, B_col);
    std::cout << "\nMatix C:\n";
    PrintMatrix <double> ((double *)HostMatC, C_row, C_col);

  }
      
  else if(n == 'c') {
    DefineCuMat<cuComplex>((cuComplex *)HostMatA, A_row, A_col);
    DefineCuMat<cuComplex>((cuComplex *)HostMatB, B_row, B_col);
    DefineCuMat<cuComplex>((cuComplex *)HostMatC, C_row, C_col);

    // printing input matrices
    std::cout << "\nMatrix A:\n";
    PrintCuMatrix <cuComplex> ((cuComplex *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintCuMatrix <cuComplex> ((cuComplex *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintCuMatrix <cuComplex> ((cuComplex *)HostMatC, C_row, C_col);
    cuComplex alpha ={(float)alpha_real, (float)alpha_imaginary}; 
    cuComplex beta ={(float)beta_real, (float)beta_imaginary};
  }
  
  else if(n =='z') {
    DefineCuMat<cuDoubleComplex>((cuDoubleComplex *)HostMatA, A_row, A_col);
    DefineCuMat<cuDoubleComplex>((cuDoubleComplex *)HostMatB, B_row, B_col);
    DefineCuMat<cuDoubleComplex>((cuDoubleComplex *)HostMatC, C_row, C_col);

    // printing input matrices
    std::cout << "\nMatrix A:\n";
    PrintCuMatrix <cuDoubleComplex> ((cuDoubleComplex *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintCuMatrix <cuDoubleComplex> ((cuDoubleComplex *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintCuMatrix <cuDoubleComplex> ((cuDoubleComplex *)HostMatC, C_row, C_col);

    cuDoubleComplex alpha_z ={(double)alpha_real, (double)alpha_imaginary}; 
    cuDoubleComplex beta_z ={(double)beta_real, (double)beta_imaginary};
  }

  else if(n == 'h') {
    DefineMat<__half>((__half *)HostMatA, A_row, A_col);
    DefineMat<__half>((__half *)HostMatB, B_row, B_col);
    DefineMat<__half>((__half *)HostMatC, C_row, C_col);

    // printing input matrices
    std::cout << "\nMatrix A:\n";
    PrintMatrix <__half> ((__half *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintMatrix <__half> ((__half *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintMatrix <__half> ((__half *)HostMatC, C_row, C_col);

  }

  // on the device
  T *DeviceMatA; // d_A - A on the device
  T *DeviceMatB; // d_B - B on the device
  T *DeviceMatC; // d_C - C on the device

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
  
  switch (n) {
    case 's' :
      std::cout << "Calling  sGemm API\n";
      clk_start = clock();
      // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
      // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
      // alpha, beta - scalars
      status = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                          B_col, A_col,(float *) &alpha_real, (float *)DeviceMatA, A_row,
                          (float *)DeviceMatB, B_row,(float *) &beta_real, (float *)DeviceMatC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sgemm kernel execution error\n");
        return EXIT_FAILURE;
      }

    clk_end = clock();
    std::cout << "API call ended\n";
  break;

  case 'd':
    std::cout << "Calling  dGemm API\n";
    clk_start = clock();
    // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
    // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
    // alpha, beta - scalars
    status = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, (double *)&alpha_real, (double *)DeviceMatA, A_row,
                        (double *)DeviceMatB, B_row, (double *) &beta_real,(double *) DeviceMatC, C_row);
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!!  Dgemm kernel execution error\n");
      return EXIT_FAILURE;
    
    }

    clk_end = clock();
    std::cout << "API call ended\n";
    break;

    case 'h': 
    std::cout << "Calling  HGemm API\n";
    clk_start = clock();
    // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
    // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
    // alpha, beta - scalars

    status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, (__half *)&alpha_real, (__half *)DeviceMatA, A_row,
                        (__half *)DeviceMatB, B_row, (__half *) &beta_real,(__half *) DeviceMatC, C_row);
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!!  Hgemm kernel execution error\n");
      return EXIT_FAILURE;
    
    }

    clk_end = clock();
    std::cout << "API call ended\n";
    break;   
    
    case 'c' :
    std::cout << "Calling  CGemm API\n";
    clk_start = clock();
    // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
    // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
    // alpha, beta - scalars

    status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, (cuComplex *)&alpha, (cuComplex *)DeviceMatA, A_row,
                        (cuComplex *)DeviceMatB, B_row, (cuComplex *) &beta,(cuComplex *) DeviceMatC, C_row);
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!!  Cgemm kernel execution error\n");
      return EXIT_FAILURE;
    
    }

    clk_end = clock();
    std::cout << "API call ended\n";
    break;
    
    case 'z' :
    std::cout << "Calling  CGemm API\n";
    clk_start = clock();
    // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
    // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
    // alpha, beta - scalars

    status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row, 
                        B_col, A_col, (cuDoubleComplex *)&alpha_z, (cuDoubleComplex *)DeviceMatA, A_row,
                        (cuDoubleComplex *)DeviceMatB, B_row, (cuDoubleComplex *) &beta_z,(cuDoubleComplex *) DeviceMatC, C_row);
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!!  Zgemm kernel execution error\n");
      return EXIT_FAILURE;
    
    }

    clk_end = clock();
    std::cout << "API call ended\n";
    break;

   
  
  }
  
  
  status = cublasGetMatrix(C_row, C_col, sizeof (*HostMatC),
                            DeviceMatC, C_row, HostMatC, C_row); // copy d_z -> C

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after Gemm operation is:\n";
  if (n == 's') {
    PrintMatrix <float> ((float *)HostMatC, C_row, C_col); 
  }

  else if (n =='d') {
      PrintMatrix <double> ((double *)HostMatC, C_row, C_col); 
  }

  else if (n == 'c')
  {
      PrintCuMatrix<cuComplex>((cuComplex *)HostMatC, C_row ,C_col);
  }

  else if (n =='z') {
      PrintCuMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatC, C_row ,C_col);
  }

  else if (n == 'h') {
      PrintMatrix <__half> ((__half *)HostMatC, C_row, C_col); 
      
  }
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
  return EXIT_SUCCESS;
  
}
  





int main (int argc, char **argv) {
  
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char  n;

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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = atof(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      n = *(argv[loop_count + 1]);
  }
 
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  //function call
  
  if (n =='s') {
    std::cout << "Calling sGemm function\n";
    Gemm <float,float> (A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
  }
  else if(n == 'd') {
    alpha_real =  (double)alpha_real;
    alpha_imaginary = (double) alpha_imaginary;
    beta_real = (double)beta_real;
    beta_imaginary = (double) beta_imaginary;
    std::cout << "Calling DGemm function\n";
    Gemm <double,double> (A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);

  }

  else if(n == 'c') {
  
    std::cout << "Calling CGemm function\n";
    Gemm <cuComplex,float> (A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);

  }

  else if(n == 'z') {
  
    std::cout << "Calling ZGemm function\n";
    Gemm <cuDoubleComplex, double> (A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);

  }
  else if(n == 'h') {
      Gemm <__half, __half> (A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
      
  }

  return EXIT_SUCCESS;
}
