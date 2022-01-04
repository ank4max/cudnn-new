# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string>
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 
#define RANDOM (rand() % 10000 * 1.00) / 100

void PrintMatrix(cuComplex* Matrix, int matriA_row, int matriA_col) {
  int row, col;
  for (row = 0; row < matriA_row; row++) {
    for (col = 0; col < matriA_col; col++) {
      std::cout << Matrix[INDEX(row, col, matriA_row)].x << "+" << Matrix[INDEX(row, col, matriA_row)].y << "*I "    ;
    }
    std::cout << "\n";
  }
}
  

int main (int argc, char **argv) {
  
  int A_row, A_col, B_row, B_col, C_row, C_col ;
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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = atof(argv[loop_count + 1]);
  }
  
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  alpha_imaginary = 0.0f;
  beta_imaginary = 0.0f;
  
  cudaError_t cudaStatus; // cudaMalloc status
  cublasStatus_t status; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context
  int row, col; // i-row index , j-col. ind.
  
  time_t clk_start, clk_end;
  // data preparation on the host
  cuComplex *HostMatA; // mxm complex matrix a on the host
  cuComplex *HostMatB; // mxn complex matrix b on the host
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
  
  
  // define the lower triangle of an mxm Hermitian matrix a in
  // lower mode column by column
   // a:
  for (col = 0; col < A_col; col++) {                 
    for (row = 0; row < A_row; row++) {                                   
      if(row >= col) {                                        
        HostMatA[INDEX(row, col, A_row)].x = RANDOM;                   
        HostMatA[INDEX(row, col, A_row)].y = 0.0f;                       
      }                                                           
    }
  }
  // print the lower triangle of a row by row
  std::cout << "lower triangle of A :\n";
  for (row = 0; row < A_row; row++){
    for (col = 0; col < A_col; col++) {
      if(row >= col) {
        std::cout << HostMatA[INDEX(row, col, A_row)].x << "+" << HostMatA[INDEX(row, col, A_row)].y << "*I "    ;                              
      }
    }
  std::cout << "\n";
  }
  // define mxn matrices b column by column
  // b,c:
  for(col = 0; col < B_col; col++) {           
    for(row = 0; row < B_row; row++) {                      
      HostMatB[INDEX(row, col, B_row)].x = RANDOM;            
      HostMatB[INDEX(row, col, B_row)].y = 0.0f;                   
                   
    }
  }
  
  // define mxn matrices c column by column 
  for(col = 0; col < C_col; col++) {           
    for(row = 0; row < C_row; row++) {                      
      HostMatC[INDEX(row, col, C_row)].x = RANDOM;              
      HostMatC[INDEX(row, col, C_row)].y = 0.0f;                 
    }
  }
  
  // print B row by row
  std::cout << "B:\n";
  PrintMatrix(HostMatB, B_row, B_col);
  
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
  // Hermitian matrix - matrix multiplication :
  // d_c =al*d_a *d_b +bet *d_c ;
  // d_a - mxm hermitian matrix ; d_b ,d_c - mxn - general matices ;
  // al ,bet - scalars
  
  clk_start = clock();
  status = cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
  A_row, B_col, &alpha, DeviceMatA, A_row, DeviceMatB, B_row, &beta, DeviceMatC, C_row);
  
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC), DeviceMatC, C_row, HostMatC, C_row); // d_c -> c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from device to host failed\n");
    return EXIT_FAILURE;
  }
 printf ("c after Chemm :\n");
 PrintMatrix(HostMatC, C_row, C_col);
  
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  //free device memory
  cudaStatus = cudaFree (DeviceMatA); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for A\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatB); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for B\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatC); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for C\n";
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
