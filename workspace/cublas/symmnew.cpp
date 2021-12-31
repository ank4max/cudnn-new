#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define FIRST_ARG "x_row"    //for comparison with command line argument and initializing value of no. of rows for x
#define SECOND_ARG "y_col"   //for comparison with command line argument and initializing value of no. of col for y 
#define THIRD_ARG "alpha"    //for comparison with command line argument and initializing value of scalar constant alpha
#define FOURTH_ARG "beta"    //for comparison with command line argument and initializing value of scalar constant beta
#define FIRST_ARG_LEN 5      // defining length for   first cmd line argument for comparison
#define SECOND_ARG_LEN 5     // defining length for  second cmd line argument for comparison
#define THIRD_ARG_LEN 5      // defining length for  third cmd line argument  for comparison
#define FOURTH_ARG_LEN 4     // defining length for  fourth cmd line argument  for comparison
#define BEGIN 1
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))    // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

//1e-9 for converting throughput in GFLOP/sec, multiplying by 2 because each multiply-add operation uses two flops and 
// then divided it by latency to get required throughput


void PrintMatrix(float* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[index(row, col, matrix_row)] << " ";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {

  //variables for dimension of matrices
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha, beta;

  //status variable declaration
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ;
  clock_t clk_start, clk_end;

  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }
  for (int loop_count = 1; loop_count < argc; loop_count++) {
    std::string str1(argv[loop_count]);
    if (!((str1.substr(BEGIN, FIRST_ARG_LEN)).compare(FIRST_ARG)))
      x_row = atoi(argv[arg] + FIRST_ARG_LEN + 1);
    else if (!((str1.substr(BEGIN, SECOND_ARG_LEN)).compare(SECOND_ARG)))
      y_col = atoi(argv[arg] + SECOND_ARG_LEN + 1);
    else if (!((str1.substr(BEGIN, THIRD_ARG_LEN)).compare(THIRD_ARG)))
      alpha = atof(argv[arg] + THIRD_ARG_LEN + 1);
    else if (!((str1.substr(BEGIN, FOURTH_ARG_LEN)).compare(FOURTH_ARG)))
      beta = atof(argv[arg] + FOURTH_ARG_LEN + 1);
  }

  x_col = x_row;
  y_row = x_row;
  z_row = x_row;
  z_col = y_col;

  float * HostMatX; // mxm matrix x on the host
  float * HostMatY; // mxn matrix y on the host
  float * HostMatZ; // mxn matrix z on the host
  
  HostMatX = new float[x_row * x_col]; // host memory for x
  HostMatY = new float[y_row * y_col]; // host memory for y
  HostMatZ = new float[z_row * z_col]; // host memory for z

  if (HostMatX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixZ)\n");
    return EXIT_FAILURE;
  }
  
  int row, col;
  // define the lower triangle of an mxm symmetric matrix x in
  // lower mode column by column
  //using rand() function to generate random numbers converted them to float 
  // and made them less than 100.
  for (col = 0; col < x_col; col++) {
    for (row = 0; row < x_row; row++) {
      if(row >=col) {
        HostMatX[INDEX(row, col, x_row)] = (rand() % 10000 * 1.00) / 100;
      }
    } 
  }
  
  // print the lower triangle of a row by row
  std::cout << "\nLower Triangle of X:\n";
  for (row = 0; row < x_row; row++) {
    for (col = 0; col < x_col; col++) {
      if (row >= col) {
        std::cout << HostMatX[INDEX(row, col, x_row)] << " ";
      }
    }
    std::cout<<"\n";
  }
  
  // define mxn matrices y and z column by column
  for (col = 0; col < y_col; col++) {
    for (row = 0; row < y_row; row++) {
      HostMatY[INDEX(row, col, y_row)] = (rand() % 10000 * 1.00) / 100;
      HostMatZ[INDEX(row, col, z_row)] = (rand() % 10000 * 1.00) / 100;
    }
  }
  
  // print y row by row
  // print z row by row
  std::cout << "\nMatriz Y:\n";
  PrintMatrix(HostMatY, y_row, y_col);
  std::cout << "\nMatriz Z:\n";
  PrintMatrix(HostMatZ, z_row, z_col);
 
  // on the device
  float *DeviceMatX; // d_x - x on the device
  float *DeviceMatY; // d_y - y on the device
  float *DeviceMatZ; // d_z - z on the device
  cudaStatus = cudaMalloc((void **) &DeviceMatX, x_row * x_col * sizeof (*HostMatX));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  // memory alloc for y
  cudaStatus = cudaMalloc((void **) &DeviceMatY, y_row * y_col * sizeof (*HostMatY));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;
  }
  // memory alloc for z
  cudaStatus = cudaMalloc((void **) &DeviceMatZ, z_row * z_col * sizeof (*HostMatZ));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Z\n";
    return EXIT_FAILURE;
  }
  
  status = cublasCreate (&handle); // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  // copy matrices from the host to the device
  status = cublasSetMatrix (x_row, x_col, sizeof (*HostMatX), HostMatX, x_row, DeviceMatX, x_row); //x -> d_x
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (y_row, y_col, sizeof (*HostMatY), HostMatY, y_row, DeviceMatY, y_row); //y -> d_y
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed\n");
    return EXIT_FAILURE;
  }
 
  status = cublasSetMatrix (z_row, z_col, sizeof (*HostMatZ), HostMatZ, z_row, DeviceMatZ, z_row); //z -> d_z
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed\n");
    return EXIT_FAILURE;
  }
 
  clk_start = clock();

  // symmetric matrix - matrix multiplication : 
  status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        y_row, y_col, &alpha, DeviceMatX, x_row, DeviceMatY,
                        y_row, &beta, DeviceMatZ, z_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();

  status = cublasGetMatrix (z_row, z_col, sizeof (*HostMatZ), DeviceMatZ, z_row, HostMatZ, z_row); // d_z -> z
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from device to host failed\n");
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz Z after Symm operation is:\n";
  PrintMatrix(HostMatZ, z_row, z_col);
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  
  cudaStatus = cudaFree (DeviceMatX); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for X\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatY); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for Y\n";
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatZ); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for Z\n";
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
// lower triangle of a:
// 11
// 12 17
// 13 18 22
// 14 19 23 26
// 15 20 24 27 29
// 16 21 25 28 30 31
// b(=c):
// 11 17 23 29
// 12 18 24 30
// 13 19 25 31
// 14 20 26 32
// 15 21 27 33
// 16 22 28 34
// c after Ssymm :
// 1122 1614 2106 2598
// 1484 2132 2780 3428
// 1740 2496 3252 4008 // c=al*a*b+bet*c
// 1912 2740 3568 4396
// 2025 2901 3777 4653
// 2107 3019 3931 4843
