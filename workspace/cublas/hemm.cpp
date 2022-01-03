# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string>
#define index(i ,j , ld ) ((( j )*( ld ))+( i ))

#define m 6 // a - mxm matrix
#define n 5 // b,c - mxn matrices
int main ( void ) {
  
  cudaError_t cudaStatus ; // cudaMalloc status
  cublasStatus_t status ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
  int i,j; // i-row index , j-col. ind.
  time_t start, end;
  // data preparation on the host
  cuComplex *HostMatX; // mxm complex matrix a on the host
  cuComplex *HostMatY; // mxn complex matrix b on the host
  cuComplex *HostMatZ; // mxn complex matrix c on the host
  HostMatX = (cuComplex *) malloc (m * m * sizeof (cuComplex)); // host memory
  // alloc for a
  HostMatY = (cuComplex *) malloc (m * n * sizeof (cuComplex)); // host memory
  // alloc for b
  HostMatZ = (cuComplex *) malloc (m * n * sizeof (cuComplex)); // host memory
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
  for (j = 0; j < m; j++) {                 // 11
    for (i = 0; i < m; i++) {                                   // 12 ,17
      if(i >=j) {                                        // 13 ,18 ,22
        HostMatX[index(i, j, m)].x = (float)ind ++;                   // 14 ,19 ,23 ,26
        HostMatX[index(i, j, m)].y = 0.0f;                       // 15 ,20 ,24 ,27 ,29
      }                                                           // 16 ,21 ,25 ,28 ,30 ,31
    }
  }
  // print the lower triangle of a row by row
  printf (" lower triangle of a:\n");
  for (i = 0; i < m; i++){
    for (j = 0; j < m; j++) {
      if(i >=j) {
        std::cout << HostMatX[index(i,j,m)].x << "+" << HostMatX[index(i,j,m)].y << "*I "    ;                              
      }
    }
  std::cout << "\n";
  }
  // define mxn matrices b,c column by column
  ind =11; // b,c:
  for(j = 0; j < n; j++) {           // 11 ,17 ,23 ,29 ,35
    for(i = 0; i < m; i++) {                      // 12 ,18 ,24 ,30 ,36
      HostMatY[index(i,j,m)].x=( float )ind;            // 13 ,19 ,25 ,31 ,37
      HostMatY[index(i,j,m)].y =0.0f;                   // 14 ,20 ,26 ,32 ,38
      HostMatZ[index(i,j,m)].x=( float )ind;              // 15 ,21 ,27 ,33 ,39
      HostMatZ[index(i,j,m)].y =0.0f;             // 16 ,22 ,28 ,34 ,40
      ind ++;
    }
  }
  // print b(=c) row by row
printf ("b,c:\n");
for (i=0;i<m;i ++){
for (j=0;j<n;j ++){
std::cout << HostMatY[index(i,j,m)].x << "+" << HostMatY[index(i,j,m)].y << "*I "    ;
}
std::cout << "\n";
}

  // on the device
  cuComplex * DeviceMatX; // d_a - a on the device
  cuComplex * DeviceMatY; // d_b - b on the device
  cuComplex * DeviceMatZ; // d_c - c on the device
  cudaStatus = cudaMalloc ((void **)& DeviceMatX , m * m * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  
  // device memory alloc for a
  cudaStatus = cudaMalloc ((void **)& DeviceMatY , n * m * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for b
  cudaStatus = cudaMalloc ((void **)& DeviceMatZ, n * m * sizeof (cuComplex));
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
  status = cublasSetMatrix (m, m, sizeof (*HostMatX) , HostMatX, m, DeviceMatX, m); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix X from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (m, n, sizeof (*HostMatY) , HostMatY, m, DeviceMatY, m); //b -> d_b
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Y from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (m, n, sizeof (*HostMatZ) , HostMatZ, m, DeviceMatZ, m); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from host to device failed \n");
    return EXIT_FAILURE;
  }
  cuComplex al ={1.0f ,0.0f}; // al =1
  cuComplex bet ={1.0f ,0.0f}; // bet =1
  // Hermitian matrix - matrix multiplication :
  // d_c =al*d_a *d_b +bet *d_c ;
  // d_a - mxm hermitian matrix ; d_b ,d_c - mxn - general matices ;
  // al ,bet - scalars
  
  start = clock();
  status = cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
  m, n, &al, DeviceMatX, m, DeviceMatY, m, &bet, DeviceMatZ, m);
  
  end = clock();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  status = cublasGetMatrix (m, n, sizeof (*HostMatZ), DeviceMatZ, m, HostMatZ, m); // d_c -> c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix Z from device to host failed\n");
    return EXIT_FAILURE;
  }
 printf ("c after Chemm :\n");
for (i=0;i<m;i ++){
for (j=0;j<n;j ++){ // print c after Chemm
std::cout << HostMatZ[index(i,j,m)].x << "+" << HostMatZ[index(i,j,m)].y << "*I "    ;;
}
std::cout << "\n";
}
  
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";
  

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
  
  free (HostMatX); // free host memory
  free (HostMatY); // free host memory
  free (HostMatZ); // free host memory
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
