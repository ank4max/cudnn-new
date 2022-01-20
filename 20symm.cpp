%%writefile max1.cpp
#include <iostream>
#include <string.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#define RANDOM (rand() % 10000 * 1.00) / 100     // to generate random values
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))  
/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

cudaError_t cudaStatus ; 
cublasStatus_t status ; 
cublasHandle_t handle ;

clock_t clk_start, clk_end;

template<class A>
void LowDef1(A* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (col = 0; col < matrix_col; col++) {
    for (row = 0; row < matrix_row; row++) {
      if(row >=col) {
        Matrix[INDEX(row, col, matrix_row)] = RANDOM;
      }
    } 
  }  
}

template<class A1>
void LowPrint1(A1 * Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      if (row >= col) {
        std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
      }
    }
    std::cout<<"\n";
  }
}


template<class Data>
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

template<class D>
void DefineMat(D* Matrix, int matrix_row, int matrix_col) {
  int row , col;  
  for (row = 0; row < matrix_row; row++) {                                              
    for (col = 0; col < matrix_col; col++) {                                                   
      Matrix[INDEX(row, col, matrix_row)] = RANDOM;                                      
    }                                                                                    
  }                                                                               
}

template<class C>
void DefineCuMat(C* Matrix, int matrix_row, int matrix_col) {
  int row, col;  
  for(col = 0; col < matrix_col; col++) {           
    for(row = 0; row < matrix_row; row++) {                      
      Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
      Matrix[INDEX(row, col, matrix_row)].y = 0.0f;              
    }
  }
}

template<class P>
void PrintCuMatrix(P* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
    }
    std::cout << "\n";
  } 
}

template<class J>
void LowDef(J* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (col = 0; col < matrix_col; col++) {                 
    for (row = 0; row < matrix_row; row++) {                                   
      if(row >= col) {                                        
        Matrix[INDEX(row, col, matrix_row)].x = RANDOM;                   
        Matrix[INDEX(row, col, matrix_row)].y = 0.0f;                       
      }                                                           
    }
  }
}

template<class K>
void LowPrint(K * Matrix, int matrix_row, int matrix_col) {
  int row, col;
  std::cout << "lower triangle of A :\n";
  for (row = 0; row < matrix_row; row++){
    for (col = 0; col < matrix_col; col++) {
      if(row >= col) {
          std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";                              
      }
    }
    std::cout << "\n";
  }
}

template<class T, class T1>
int Symm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T1 alpha_real, T1 alpha_imaginary, T1 beta_real, T1 beta_imaginary, char mode) {
  
  int row, col;
  clk_start = 0;
  clk_end = 0;
  cuComplex alpha;
  cuComplex beta;
  cuDoubleComplex alpha_z; 
  cuDoubleComplex beta_z;

  //Host Matrices  Declaration
  T *HostMatA; 
  T *HostMatB;
  T *HostMatC; 
  
  //host Memory Allocation for Matrices
  HostMatA = new T[A_row * A_col]; 
  HostMatB = new T[B_row * B_col]; 
  HostMatC = new T[C_row * C_col]; 
  
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


  if(mode == 'S') {
      
    // define the lower triangle of an mxm symmetric matrix A in
    // lower mode column by column
    //using rand() function to generate random numbers converted them to float 
    // and made them less than 100.
    LowDef1<float>((float *)HostMatA, A_row, A_col);
    DefineMat<float>((float *)HostMatB, B_row, B_col);
    DefineMat<float>((float *)HostMatC, C_row, C_col);

    // printing input matrices
    // print the lower triangle of a row by row
    std::cout << "\nLower Triangle of A:\n";
    LowPrint1<float>((float *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintMatrix <float> ((float *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintMatrix <float> ((float *)HostMatC, C_row, C_col);
  }

  else if (mode == 'D') {
    // define the lower triangle of an mxm symmetric matrix A in
    // lower mode column by column
    //using rand() function to generate random numbers converted them to float 
    // and made them less than 100.
    LowDef1<double>(( double *)HostMatA, A_row, A_col);
    DefineMat<double>((double *)HostMatB, B_row, B_col);
    DefineMat<double>((double *)HostMatC, C_row, C_col);

    // printing input matrices
    // print the lower triangle of a row by row
    std::cout << "\nLower Triangle of A:\n";
    LowPrint1<double>((double *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintMatrix <double> ((double *)HostMatB, B_row, B_col);
    std::cout << "\nMatix C:\n";
    PrintMatrix <double> ((double *)HostMatC, C_row, C_col);
  }
      
  else if(mode == 'C') {
    // define the lower triangle of a row by row by row
    // lower mode column by column
    // a:
    LowDef<cuComplex>((cuComplex *)HostMatA, A_row, A_col);
    DefineCuMat<cuComplex>((cuComplex *)HostMatB, B_row, B_col);
    DefineCuMat<cuComplex>((cuComplex *)HostMatC, C_row, C_col);

    // printing input matrices
    // print the lower triangle of a row by row
    LowPrint<cuComplex>((cuComplex *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintCuMatrix <cuComplex> ((cuComplex *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintCuMatrix <cuComplex> ((cuComplex *)HostMatC, C_row, C_col);
    
    //initializing values for alpha  and beta
    cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary}; 
    cuComplex beta = {(float)beta_real, (float)beta_imaginary};
  }

  else if(mode == 'Z') {
    // define the lower triangle of a row by row by row
    // lower mode column by column
    // a:
    LowDef<cuDoubleComplex>((cuDoubleComplex *)HostMatA, A_row, A_col);
    DefineCuMat<cuDoubleComplex>((cuDoubleComplex *)HostMatB, B_row, B_col);
    DefineCuMat<cuDoubleComplex>((cuDoubleComplex *)HostMatC, C_row, C_col);

    // printing input matrices
    // print the lower triangle of a row by row
    LowPrint<cuDoubleComplex>((cuDoubleComplex *)HostMatA, A_row, A_col);
    std::cout << "\nMatrix B:\n";
    PrintCuMatrix <cuDoubleComplex> ((cuDoubleComplex *)HostMatB, B_row, B_col);
    std::cout << "\nMatrix C:\n";
    PrintCuMatrix <cuDoubleComplex> ((cuDoubleComplex *)HostMatC, C_row, C_col);
    
    //initializing values for alpha  and beta
    cuDoubleComplex alpha_z = {(float)alpha_real, (float)alpha_imaginary}; 
    cuDoubleComplex beta_z = {(float)beta_real, (float)beta_imaginary};
  }
  
  // Memory allocation on the device
  T *DeviceMatA; 
  T *DeviceMatB; 
  T *DeviceMatC; 

  cudaStatus = cudaMalloc((void **)&DeviceMatA, A_row * A_col * sizeof (*HostMatA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceMatB, B_row * B_col * sizeof (*HostMatB));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceMatC, C_row * C_col * sizeof (*HostMatC));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
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
  
  switch (mode) {
    case 'S':
      std::cout << "Calling  ssymm API\n";
      clk_start = clock();

      // symmetric matrix - matrix multiplication : 
      status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        B_row, B_col, (float *)&alpha_real, (float *)DeviceMatA, A_row, (float *)DeviceMatB,
                        B_row, (float *) &beta_real, (float *)DeviceMatC, C_row);
  
      clk_end = clock();
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! kernel execution error\n");
        return EXIT_FAILURE;
      }
      std::cout << "API call ended\n";
      break;

    case 'D':
      std::cout << "Calling  dsymm API\n";
      clk_start = clock();
      // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
      // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
      // alpha, beta - scalars
      status = cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        B_row, B_col, (double *)&alpha_real, (double *)DeviceMatA, A_row, (double *)DeviceMatB,
                        B_row, (double *)&beta_real, (double *)DeviceMatC, C_row);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dsymm kernel execution error\n");
        return EXIT_FAILURE;
      }
      clk_end = clock();
      std::cout << "API call ended\n";
      break;

    case 'C':
      std::cout << "Calling  Csymm API\n";
      clk_start = clock();
      // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
      // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
      // alpha, beta - scalars
      status = cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        B_row, B_col, (cuComplex *)&alpha, (cuComplex *)DeviceMatA, A_row, (cuComplex *)DeviceMatB,
                        B_row, (cuComplex *)&beta, (cuComplex*)DeviceMatC, C_row);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Csymm kernel execution error\n");
        return EXIT_FAILURE;
      }
      clk_end = clock();
      std::cout << "API call ended\n";
      break;
    
    case 'Z':
      std::cout << "Calling  Zsymm API\n";
      clk_start = clock();
      // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
      // d_A -mxk matrix , d_B -kxn matrix , d_C -mxn matrix
      // alpha, beta - scalars
      status = cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        B_row, B_col, (cuDoubleComplex *)&alpha_z, (cuDoubleComplex *)DeviceMatA, A_row, (cuDoubleComplex *)DeviceMatB,
                        B_row, (cuDoubleComplex *)&beta_z, (cuDoubleComplex*)DeviceMatC, C_row);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zsymm kernel execution error\n");
        return EXIT_FAILURE;
      }
      clk_end = clock();
      std::cout << "API call ended\n";
      break;
  }


  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC), DeviceMatC, C_row, HostMatC, C_row); // d_z -> z
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from device to host failed\n");
    return EXIT_FAILURE;
  }

  std::cout << "\nMatriz C after Symm operation is:\n";
  if (mode == 'S') {
    PrintMatrix<float>((float *)HostMatC, C_row, C_col); 
  }

  else if (mode == 'D') {
      PrintMatrix <double>((double *)HostMatC, C_row, C_col); 
  }

  else if (mode == 'C') {
    PrintCuMatrix<cuComplex>((cuComplex *)HostMatC, C_row ,C_col);
  }

  else if (mode == 'Z') {
    PrintCuMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatC, C_row ,C_col);
  }

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " <<THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  cudaStatus = cudaFree (DeviceMatA); // free device memory
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatB); // free device memory
  if(cudaStatus != cudaSuccess) {
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
 
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  //function call
  
  if (n =='S') {
    std::cout << "Calling ssymm function\n";
    Symm<float, float>(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
  }
  else if(n == 'D') {
    alpha_real =  (double)alpha_real;
    alpha_imaginary = (double) alpha_imaginary;
    beta_real = (double)beta_real;
    beta_imaginary = (double) beta_imaginary;
    std::cout << "Calling Dsymm function\n";
    Symm<double, double>(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
  }

  else if(n == 'C') {
    std::cout << "Calling CGemm function\n";
    Symm<cuComplex, float>(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
  }

  else if(n == 'Z') {
    std::cout << "Calling CGemm function\n";
    Symm<cuDoubleComplex, float>(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary, n);
  }

  return EXIT_SUCCESS;
}


















