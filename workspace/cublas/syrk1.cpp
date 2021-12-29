#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
# include "cublas_v2.h"

# define IDX2C(i ,j , ld ) ((( j )*( ld ))+( i ))
# define n 6 // a - nxk matrix
# define k 4 // c - nxn matrix

int main ( void ) {
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ; 
  int i,j; // i-row index , j- column index
  
  float *HostMatX;                   // nxk matrix a on the host
  float *HostMatY;                   // nxn matrix c on the host
  HostMatX = (float *) malloc (n*k* sizeof (float)); // host memory for a
  HostMatY = (float *) malloc (n*n* sizeof (float)); // host memory for c
  
  if (HostMatX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an nxn symmetric matrix c
  // column by column
  int ind =11; // c:
  for(j = 0; j < n; j++) { // 11
    for(i = 0; i < n; i++) { // 12 ,17
      if(i >= j) { // 13 ,18 ,22
        HostMatY[ IDX2C (i,j,n )] = (float)ind ++; // 14 ,19 ,23 ,26
      } // 15 ,20 ,24 ,27 ,29
    } // 16 ,21 ,25 ,28 ,30 ,31
  }
  
  // print the lower triangle of c row by row
  printf (" lower triangle of c:\n");
  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++) {
      if(i >= j) {
        printf (" %5.0f",HostMatY[ IDX2C (i,j,n )]);
      }
    }
    printf ("\n");
  }
  
  // define an nxk matrix a column by column
  ind =11; // a:
  for(j = 0; j < k; j++) { // 11 ,17 ,23 ,29
    for(i = 0; i < n; i++) { // 12 ,18 ,24 ,30
      HostMatX[ IDX2C (i,j,n )] = ( float )ind; // 13 ,19 ,25 ,31
      ind ++; // 14 ,20 ,26 ,32
    } // 15 ,21 ,27 ,33
  } // 16 ,22 ,28 ,34

  printf ("a:\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < k; j++) {
      printf (" %5.0f",HostMatX[ IDX2C (i,j,n )]); // print a row by row
    }
    printf ("\n");
  }
  
  // on the device
  float * DeviceMatX; // d_a - a on the device
  float * DeviceMatY; // d_c - c on the device

  
  cudaStatus = cudaMalloc((void **)& DeviceMatX, n * k * sizeof (*HostMatX)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X\n";
    return EXIT_FAILURE;
  }
  // memory alloc for a
  cudaStatus = cudaMalloc((void **)& DeviceMatY, n*n* sizeof (*HostMatY)); // device
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y\n";
    return EXIT_FAILURE;
  }
  // memory alloc for c
  status = cublasCreate (& handle); // initialize CUBLAS context
// copy matrices from the host to the device
stat = cublasSetMatrix (n,k, sizeof (*a) ,a,n,d_a ,n); //a -> d_a
stat = cublasSetMatrix (n,n, sizeof (*c) ,c,n,d_c ,n); //c -> d_c
float al =1.0f; // al =1
float bet =1.0f; // bet =1
// symmetric rank -k update : c = al*d_a *d_a ^T + bet *d_c ;
// d_c - symmetric nxn matrix , d_a - general nxk matrix ;
// al ,bet - scalars
stat=cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,
n,k,&al,d_a,n,&bet,d_c,n);
stat = cublasGetMatrix (n,n, sizeof (*c) ,d_c ,n,c,n); // d_c -> c
printf (" lower triangle of updated c after Ssyrk :\n");
for(i=0;i<n;i ++){
for(j=0;j<n;j ++){
if(i >=j) // print the lower triangle
printf (" %7.0f",c[ IDX2C (i,j,n )]); // of c after Ssyrk
}
printf ("\n");
}
cudaFree (d_a ); // free device memory
cudaFree (d_c ); // free device memory
cublasDestroy ( handle ); // destroy CUBLAS context
free (a); // free host memory
free (c); // free host memory
return EXIT_SUCCESS ;
}
