# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string>
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100      // to generate random values
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 
/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */

void PrintMatrix(cuComplex* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
    }
    std::cout << "\n";
  }
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
  // reading cmd line arguments
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
  
  //initializing values for C matrix
  C_row = A_row;
  C_col = A_row;
  
  // creating cublas handle
  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle; 
  int row, col;
  clock_t clk_start, clk_end;
  
  // allocating memory for matrices on host
  cuComplex *HostMatA; 
  cuComplex *HostMatC; 
  HostMatA = new cuComplex[A_row * A_col]; 
  HostMatC = new cuComplex[C_row * C_col]; 
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix A)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix C)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an nxn Hermitian matrix c in
  // lower mode column by column ;
  // setting up values for matrices
  // using RANDOM macro to generate random numbers between 0 - 100
  for(col = 0; col < C_col; col++) {           
    for(row = 0; row < C_row; row++) {            
      if(row >= col) {                                  
        HostMatC[INDEX(row, col, C_row)].x = RANDOM;
        HostMatC[INDEX(row, col, C_row)].y = 0.0f;                 
      }                                                           
    }
  }
  
  // print the lower triangle of c row by row
  std::cout << "lower triangle of C :\n";
  for (row = 0; row < C_row; row++){
    for (col = 0; col < C_col; col++) {
      if(row >= col) {
        std::cout << HostMatC[INDEX(row, col, C_row)].x << "+" << HostMatC[INDEX(row, col, C_row)].y << "*I ";                              
      }
    }
  std::cout << "\n";
  }
  
  // setting up values for matrices
  // using RANDOM macro to generate random numbers between 0 - 100
  //defining a matrix A 
  for(col = 0; col < A_col; col++) {           
    for(row = 0; row < A_row; row++) {                      
      HostMatA[INDEX(row, col, A_row)].x = RANDOM;            
      HostMatA[INDEX(row, col, A_row)].y = 0.0f;                            
    }
  }
  
  // print A row by row
  std::cout << "A:\n";
  PrintMatrix(HostMatA, A_row, A_col);
 
  // allocating memory for matrices on device using cudaMalloc
  cuComplex *DeviceMatA;  
  cuComplex *DeviceMatC;  
  cudaStatus = cudaMalloc ((void **)& DeviceMatA, A_row * A_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **)& DeviceMatC, C_row * C_col * sizeof (cuComplex));
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
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA), HostMatA, A_row, DeviceMatA, A_row); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC), HostMatC, C_row, DeviceMatC, C_row); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  // rank -k update of a Hermitian matrix :
  // d_c =al*d_a *d_a ^H +bet *d_c
  // d_c - nxn , Hermitian matrix ; d_a - nxk general matrix ;
  // al ,bet - scalars
  
  // start variable to store time
  clk_start = clock();
  status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  A_row, A_col, &alpha, DeviceMatA, A_row, &beta, DeviceMatC, C_row);
  
  // end variable to store time
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  // getting the final output
  status = cublasGetMatrix (C_row, C_col, sizeof (*HostMatC), DeviceMatC, C_row, HostMatC, C_row); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from device to host failed\n");
    return EXIT_FAILURE;
  }
  
  // Matrix output
  std::cout << "Lower triangle of c after Cherk :\n";
  for(row = 0; row < C_row; row++) {
    for(col = 0; col < C_col; col ++) { // print c after Cherk
      if(row >= col) {
        std::cout << HostMatC[INDEX(row, col, C_row)].x << "+" << HostMatC[INDEX(row, col, C_row)].y << "*I ";
      }
    }
    std::cout << "\n";
  }
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  //free device memory
  cudaStatus = cudaFree (DeviceMatA); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for A\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatC); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for C\n";
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
  delete[] HostMatC; 
  return EXIT_SUCCESS ;
}
// lower triangle of c:
// 11+ 0*I
// 12+ 0*I 17+ 0*I
// 13+ 0*I 18+ 0*I 22+ 0*I
// 14+ 0*I 19+ 0*I 23+ 0*I 26+ 0*I
// 15+ 0*I 20+ 0*I 24+ 0*I 27+ 0*I 29+ 0*I
// 16+ 0*I 21+ 0*I 25+ 0*I 28+ 0*I 30+ 0*I 31+ 0*I
// a:
// 11+ 0*I 17+ 0*I 23+ 0*I 29+ 0*I 35+ 0*I
// 12+ 0*I 18+ 0*I 24+ 0*I 30+ 0*I 36+ 0*I
// 13+ 0*I 19+ 0*I 25+ 0*I 31+ 0*I 37+ 0*I
// 14+ 0*I 20+ 0*I 26+ 0*I 32+ 0*I 38+ 0*I
// 15+ 0*I 21+ 0*I 27+ 0*I 33+ 0*I 39+ 0*I
// 16+ 0*I 22+ 0*I 28+ 0*I 34+ 0*I 40+ 0*I
// lower triangle of c after Cherk :
// 3016+0* I
// 3132+0* I 3257+0* I
// 3248+0* I 3378+0* I 3507+0* I // c=a*a^H +c
// 3364+0* I 3499+0* I 3633+0* I 3766+0* I
// 3480+0* I 3620+0* I 3759+0* I 3897+0* I 4034+0* I
// 3596+0* I 3741+0* I 3885+0* I 4028+0* I 4170+0* I 4311+0* I
