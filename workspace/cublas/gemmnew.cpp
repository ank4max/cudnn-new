#include <iostream>
#include <string>
#include "cublas.h"
#include "cublas_v2.h"

#define FIRST_ARG "x_row"    //for comparison with command line argument and initializing value of no. of rows for x
#define SECOND_ARG "x_col"   //for comparison with command line argument and initializing value of no. of col for x
#define THIRD_ARG "y_col"    //for comparison with command line argument and initializing value of no. of col for y
#define FOURTH_ARG "alpha"   //for comparison with command line argument and initializing value of scalar constant alpha
#define FIFTH_ARG "beta"     //for comparison with command line argument and initializing value of scalar constant beta
#define LEN_ARG_FIRST 5      // defining length for   first cmd line argument for comparison
#define LEN_ARG_SECOND 5     // defining length for  second cmd line argument for comparison
#define LEN_ARG_THIRD 5      // defining length for  third cmd line argument  for comparison
#define LEN_ARG_FOURTH 5     // defining length for  fourth cmd line argument for comparison
#define LEN_ARG_FIFTH 4      // defining length for  fifth cmd line argument for comparison
#define BEGIN 1              
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

//1e-9 for converting throughput in GFLOP/sec, multiplying by 2 because each multiply-add operation uses two flops and 
// then divided it by latency to get required throughput

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
  
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha, beta;

  std::cout << std::endl;
  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }

  // reading cmd line arguments
  for (int loop_count = 1; loop_count < argc; loop_count++) {
           std::string str1(argv[loop_count]);  
    if (!((str1.substr(BEGIN, LEN_ARG_FIRST)).compare(FIRST_ARG)))
      x_row = atoi(argv[loop_count] + LEN_ARG_FIRST + 1);
      
    else if (!((str1.substr(BEGIN, LEN_ARG_SECOND)).compare(SECOND_ARG)))
      x_col = atoi(argv[loop_count] + LEN_ARG_SECOND + 1);

    else if (!((str1.substr(BEGIN, LEN_ARG_THIRD)).compare(THIRD_ARG)))
      y_col = atoi(argv[loop_count] + LEN_ARG_THIRD + 1);

    else if (!((str1.substr(BEGIN, LEN_ARG_FOURTH)).compare(FOURTH_ARG)))
      alpha = atof(argv[loop_count] + LEN_ARG_FOURTH + 1);

    else if (!((str1.substr(BEGIN, LEN_ARG_FIFTH)).compare(FIFTH_ARG)))
      beta = atof(argv[loop_count] + LEN_ARG_FIFTH + 1);
  }
 
  y_row = x_col;
  z_row = x_row;
  z_col = y_col;
  
  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle;

  clock_t clk_start, clk_end;   
 
  float *HostMatX; // mxk matrix x on the host
  float *HostMatY; // kxn matrix y on the host
  float *HostMatZ; // mxn matrix z on the host
  
  HostMatX = new float[x_row * x_col]; // host memory for x
  HostMatY = new float[y_row * y_col]; // host memory for y
  HostMatZ = new float[z_row * z_col]; // host memory for z
  
  if (HostMatX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixZ)\n");
    return EXIT_FAILURE;
  }

  int row, col;
  
  // define an mxk matrix x column by column
  //using rand() function to generate random numbers converted them to float 
  // and made them less than 100.
  for (row = 0; row < x_row; row++) {                                              
    for (col = 0; col < x_col; col++) {                                                   
      HostMatX[INDEX(row, col, x_row)] = (rand() % 10000 * 1.00) / 100;                                      
    }                                                                                    
  }                                                                               
                                                                               
  // define a kxn matrix y column by column
  //using rand() function to generate random numbers converted them to float 
  // and made them less than 100.
  for (row = 0; row < y_row; row++) {                                      
    for (col = 0; col < y_col; col++) {                                                
      HostMatY[INDEX(row, col, y_row)] = (rand() % 10000 * 1.00) / 100;                                           
    }                                                                         
  }                                                       
  
  // define an mxn matrix z column by column
  //using rand() function to generate random numbers converted them to float 
  // and made them less than 100.
  for (row = 0; row < z_row; row++) {                             
    for (col = 0; col < z_col; col++) {                                        
      HostMatZ[INDEX(row, col, z_row)] = (rand() % 10000 * 1.00) / 100;                 
    }                                                                  
  }
  
  // printing input matrices
  std::cout << "\nMatriz X:\n";
  PrintMatrix(HostMatX, x_row, x_col);
  std::cout << "\nMatriz Y:\n";
  PrintMatrix(HostMatY, y_row, y_col);
  std::cout << "\nMatriz Z:\n";
  PrintMatrix(HostMatZ, z_row, z_col);                                                           
  
  // on the device
  float *DeviceMatX; // d_x - x on the device
  float *DeviceMatY; // d_y - y on the device
  float *DeviceMatZ; // d_z - z on the device

  cudaStatus = cudaMalloc ((void **) &DeviceMatX , x_row * x_col * sizeof (*HostMatX));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatY , y_row * y_col * sizeof (*HostMatY));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatZ , z_row * z_col * sizeof (*HostMatZ));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Z " << std::endl;
    return EXIT_FAILURE;   
  }
  
  status = cublasCreate (&handle);      // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // copy matrices from the host to the device
  status = cublasSetMatrix (x_row, x_col, sizeof (*HostMatX), HostMatX, x_row, DeviceMatX, x_row); // x -> d_x
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetMatrix (y_row, y_col, sizeof (*HostMatY), HostMatY, y_row, DeviceMatY, y_row); // y -> d_y
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (z_row, z_col, sizeof (*HostMatZ), HostMatZ, z_row, DeviceMatZ, z_row); // z -> d_z
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed\n");
    return EXIT_FAILURE;
  }
  

  clk_start = clock();

  // matrix - matrix multiplication : d_z = alpha * d_x * d_y + beta * d_z
  // d_x -mxk matrix , d_y -kxn matrix , d_z -mxn matrix
  // alpha, beta - scalars
  status = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, x_row, 
                        y_col, x_col, &alpha, DeviceMatX, x_row,
                        DeviceMatY, y_row, &beta, DeviceMatZ, z_row);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();
  
  status = cublasGetMatrix (z_row, z_col, sizeof (*HostMatZ),
                            DeviceMatZ, z_row, HostMatZ, z_row); // copy d_z -> z

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix Z from device\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz Z after Gemm operation is:\n";
  PrintMatrix(HostMatZ, z_row, z_col); 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " <<THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Y" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatZ); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Z" << std::endl;
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy (handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }

  delete[] HostMatX; // free host memory
  delete[] HostMatY; // free host memory
  delete[] HostMatZ; // free host memory

  return EXIT_SUCCESS ;
}
// a:
// 11 17 23 29 35
// 12 18 24 30 36
// 13 19 25 31 37
// 14 20 26 32 38
// 15 21 27 33 39
// 16 22 28 34 40
// b:
// 11 16 21 26
// 12 17 22 27
// 13 18 23 28
// 14 19 24 29
// 15 20 25 30
// c:
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34
// c after Sgemm :
// 1566 2147 2728 3309
// 1632 2238 2844 3450
// 1698 2329 2960 3591 // c=al*a*b+bet *c
// 1764 2420 3076 3732
// 1830 2511 3192 3873
// 1896 2602 3308 4014
