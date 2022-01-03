# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string>
#define FIRST_ARG "x_row"    //for comparison with command line argument and initializing value of no. of rows for x
#define SECOND_ARG "y_col"    //for comparison with command line argument and initializing value of no. of col for y
#define THIRD_ARG "alpha_real"   //for comparison with command line argument and initializing value of scalar constant alpha
#define FOURTH_ARG "alpha_imaginary"     //for comparison with command line argument and initializing value of scalar constant beta
#define FIFTH_ARG "beta_real"
#define SIXTH_ARG "beta_imaginary"
#define LEN_ARG_FIRST 5      // defining length for   first cmd line argument for comparison
#define LEN_ARG_SECOND 5     // defining length for  second cmd line argument for comparison
#define LEN_ARG_THIRD 10      // defining length for  third cmd line argument  for comparison
#define LEN_ARG_FOURTH 15     // defining length for  fourth cmd line argument for comparison
#define LEN_ARG_FIFTH 9      // defining length for  fifth cmd line argument for comparison
#define LEN_ARG_SIXTH 14
#define BEGIN 1              
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

int main (int argc, char **argv) {
  
  
  
  
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
  int row, col; // i-row index , j-col. ind.
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha_real, alpha_imaginary, beta_real, beta_imaginary;

  std::cout << std::endl;
  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }
  
  // reading cmd line arguments
  for (int loop_count = 1; loop_count < argc; loop_count++) {
           std::string str(argv[loop_count]);  
    if (!((str.substr(BEGIN, LEN_ARG_FIRST)).compare(FIRST_ARG)))
      x_row = atoi(argv[loop_count] + LEN_ARG_FIRST + 1);
      
    else if (!((str.substr(BEGIN, LEN_ARG_SECOND)).compare(SECOND_ARG)))
      y_col = atoi(argv[loop_count] + LEN_ARG_SECOND + 1);

    else if (!((str.substr(BEGIN, LEN_ARG_THIRD)).compare(THIRD_ARG)))
      alpha_real = atof(argv[loop_count] + LEN_ARG_THIRD + 1);

    else if (!((str.substr(BEGIN, LEN_ARG_FOURTH)).compare(FOURTH_ARG)))
      alpha_imaginary = atof(argv[loop_count] + LEN_ARG_FOURTH + 1);

    else if (!((str.substr(BEGIN, LEN_ARG_FIFTH)).compare(FIFTH_ARG)))
      beta_real = atof(argv[loop_count] + LEN_ARG_FIFTH + 1);
    
    else if (!((str.substr(BEGIN, LEN_ARG_SIXTH)).compare(SIXTH_ARG)))
      beta_real = atof(argv[loop_count] + LEN_ARG_SIXTH + 1);
    
  }
  
  x_col = x_row;
  y_row = x_col;
  z_row = x_row;
  z_col = y_col;
  
  time_t clk_start, clk_end;
  // data preparation on the host
  cuComplex *HostMatX; // mxm complex matrix a on the host
  cuComplex *HostMatY; // mxn complex matrix b on the host
  cuComplex *HostMatZ; // mxn complex matrix c on the host
  HostMatX = new cuComplex[x_row * x_col]; // host memory
  // alloc for x
  HostMatY = new cuComplex[y_row * col]; // host memory
  // alloc for b
  HostMatZ = new cuComplex[z_row * z_col]; // host memory
  // alloc for c
  
  if (HostMatX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix X)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix Y)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix Z)\n");
    return EXIT_FAILURE;
  }
  
  
  // define the lower triangle of an mxm Hermitian matrix a in
  // lower mode column by column
  int ind =11; // a:
  for (col = 0; col < x_col; col++) {                 // 11
    for (row = 0; row < x_row; row++) {                                   // 12 ,17
      if(row >= col) {                                        // 13 ,18 ,22
        HostMatX[INDEX(row, col, x_row)].x = (float)ind ++;                   // 14 ,19 ,23 ,26
        HostMatX[INDEX(row, col, x_row)].y = 0.0f;                       // 15 ,20 ,24 ,27 ,29
      }                                                           // 16 ,21 ,25 ,28 ,30 ,31
    }
  }
  // print the lower triangle of a row by row
  std::cout << " lower triangle of X :\n" ;
  for (row = 0; row < x_row; row++){
    for (col = 0; col < x_col; col++) {
      if(row >= col) {
        std::cout << HostMatX[INDEX(row, col, x_row)].x << "+" << HostMatX[INDEX(row, col, x_row)].y << "*I "    ;                              
      }
    }
  std::cout << "\n";
  }
  // define mxn matrices b,c column by column
  ind =11; // b,c:
  for(col = 0; col < y_col; col++) {           // 11 ,17 ,23 ,29 ,35
    for(row = 0; row < y_row; row++) {                      // 12 ,18 ,24 ,30 ,36
      HostMatY[INDEX(row, col, y_row)].x = (float)ind;            // 13 ,19 ,25 ,31 ,37
      HostMatY[INDEX(row, col, y_row)].y =0.0f;                   // 14 ,20 ,26 ,32 ,38
                   
      ind ++;
    }
  }
  
  // define mxn matrices Z column by column
  ind =11; 
  for(col = 0; col < z_col; col++) {           
    for(row = 0; row < z_row; row++) {                      
      HostMatZ[INDEX(row, col, z_row)].x = (float)ind;              
      HostMatZ[INDEX(row, col, z_row)].y = 0.0f;             
      ind ++;
    }
  }
 // print b(=c) row by row
  printf ("b,c:\n");
  for (row = 0; row < z_row; row++) {
    for (col = 0; col < z_col; col++) {
      std::cout << HostMatZ[index(row, col, z_row)].x << "+" << HostMatZ[index(row, col, z_row)].y << "*I "    ;
    }
    std::cout << "\n";
  }

  // on the device
  cuComplex * DeviceMatX; // d_a - a on the device
  cuComplex * DeviceMatY; // d_b - b on the device
  cuComplex * DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc ((void **)& DeviceMatX , x_row * x_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  
  // device memory alloc for a
  cudaStatus = cudaMalloc ((void **)& DeviceMatY , y_row * y_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for b
  cudaStatus = cudaMalloc ((void **)& DeviceMatZ, z_row * z_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Z\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for c
  
  status = cublasCreate (& handle);  // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  // copy matrices from the host to the device
  status = cublasSetMatrix (x_row, x_col, sizeof (*HostMatX) , HostMatX, x_row, DeviceMatX, x_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (y_row, y_col, sizeof (*HostMatY) , HostMatY, y_row, DeviceMatY, y_row); //b -> d_b
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (z_row, z_col, sizeof (*HostMatZ) , HostMatZ, z_row, DeviceMatZ, z_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed \n");
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
  x_row, y_col, &alpha, DeviceMatX, x_row, DeviceMatY, y_row, &beta, DeviceMatZ, z_row);
  
  clk_end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  status = cublasGetMatrix (z_row, z_col, sizeof (*HostMatZ), DeviceMatZ, z_row, HostMatZ, z_row); // d_c -> c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from device to host failed\n");
    return EXIT_FAILURE;
  }
 printf ("c after Chemm :\n");
 for (row = 0; row < z_row; row++) {
   for (col = 0; col < z_col; col++) { // print c after Chemm
     std::cout << HostMatZ[index(row, col, z_row)].x << "+" << HostMatZ[index(row, col, z_row)].y << "*I "    ;
   }
   std::cout << "\n";
 }
  
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
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
// 11+ 0*I
// 12+ 0*I 17+ 0*I
// 13+ 0*I 18+ 0*I 22+ 0*I
// 14+ 0*I 19+ 0*I 23+ 0*I 26+ 0*I
// 15+ 0*I 20+ 0*I 24+ 0*I 27+ 0*I 29+ 0*I
// 16+ 0*I 21+ 0*I 25+ 0*I 28+ 0*I 30+ 0*I 31+ 0*I
// b,c:
// 11+ 0*I 17+ 0*I 23+ 0*I 29+ 0*I 35+ 0*I
// 12+ 0*I 18+ 0*I 24+ 0*I 30+ 0*I 36+ 0*I
// 13+ 0*I 19+ 0*I 25+ 0*I 31+ 0*I 37+ 0*I
// 14+ 0*I 20+ 0*I 26+ 0*I 32+ 0*I 38+ 0*I
// 15+ 0*I 21+ 0*I 27+ 0*I 33+ 0*I 39+ 0*I
// 16+ 0*I 22+ 0*I 28+ 0*I 34+ 0*I 40+ 0*I
// c after Chemm :
// 1122+0* I 1614+0* I 2106+0* I 2598+0* I 3090+0* I //
// 1484+0* I 2132+0* I 2780+0* I 3428+0* I 4076+0* I //
// 1740+0* I 2496+0* I 3252+0* I 4008+0* I 4764+0* I // c=a*b+c
// 1912+0* I 2740+0* I 3568+0* I 4396+0* I 5224+0* I //
// 2025+0* I 2901+0* I 3777+0* I 4653+0* I 5529+0* I //
// 2107+0* I 3019+0* I 3931+0* I 4843+0* I 5755+0* I //
