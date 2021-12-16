# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"

int main ( int argc, char **argv ) {
  
  //initializing size of arrays and creating cublas handle
  int n = atoi(argv[1]);
  cudaError_t cudaStat ;       
  cublasStatus_t stat ;        
  cublasHandle_t handle ;      
  int j;                       
  
  //pointers x  and y pointing to matrices in host memory 
  float * x;                   
  float * y;                   
  x = ( float *) malloc (n* sizeof (*x));   
  y = ( float *) malloc (n* sizeof (*y));    
  
  //setting up values in the matrices for both x and y matrices
  for (j = 0; j < n; j++) {
    x[j] = ( float )j;          
  }
  
  for (j = 0; j < n; j++) {
    y[j] = ( float )j; // y={0 ,1 ,2 ,3 ,4 ,5}
  }
  
  //printing x which has same values stored as y matrix
  printf ("x,y:\n");
  for (j = 0; j < n; j++) {
    printf (" %2.0f,", x[j]); // print x,y
  }
  printf ("\n");

  // pointers for allocating memory for matrices on device
  float * d_x; 
  float * d_y;
  
  //using cudamalloc for allocating memory on device as same as matrices on host
  cudaStat = cudaMalloc (( void **)& d_x, n* sizeof (*x)); 
  cudaStat = cudaMalloc (( void **)& d_y, n* sizeof (*y)); 
  
  // setting values of matrices on device same as that of matrices in host
  stat = cublasCreate (& handle ); 
  stat = cublasSetVector (n, sizeof (*x), x, 1, d_x, 1); 
  stat = cublasSetVector (n, sizeof (*y), y, 1, d_y, 1); 

  //scalar quantity that will be multiplied with values of matrix x and then result will be added to y for final output.
  float al =2.0; // al =2
  
  //performing saxpy operation
  stat = cublasSaxpy (handle, n, &al, d_x, 1, d_y, 1);
  
  //getting the final output in d_y and then cpying that output to y
  stat = cublasGetVector (n, sizeof ( float ), d_y, 1, y, 1); 
  
  //printing the final output
  printf ("y after Saxpy :\n"); 
  for (j = 0; j < n; j++) {
    printf (" %2.0f,", y[j]);
  }

  printf ("\n");
  //free device memory
  cudaFree (d_x ); 
  cudaFree (d_y ); 
  
  //destroying cublas handle
  cublasDestroy ( handle ); 
  
  //freeing host memory 
  free (x); 
  free (y); 
  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// 2*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
