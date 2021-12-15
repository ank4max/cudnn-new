# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"

int main ( int argc, char **argv ) {
int n = atoi(argv[1]);
cudaError_t cudaStat ;       // cudaMalloc status
cublasStatus_t stat ;        // CUBLAS functions status
cublasHandle_t handle ;      // CUBLAS context
int j;                       // index of elements
  
float * x;                   // n- vector on the host
float * y;                   // n- vector on the host
x = ( float *) malloc (n* sizeof (*x));   // host memory alloc for x
for(j = 0;j < n;j++) {
  x[j] = ( float )j;          // x={0 ,1 ,2 ,3 ,4 ,5}
}
y = ( float *) malloc (n* sizeof (*y));    // host memory alloc for y
for(j = 0;j < n;j++) {
  y[j] = ( float )j; // y={0 ,1 ,2 ,3 ,4 ,5}
}
printf ("x,y:\n");
for(j = 0;j < n;j++) {
  printf (" %2.0f,",x[j]); // print x,y
}
printf ("\n");
// on the device
float * d_x; // d_x - x on the device
float * d_y; // d_y - y on the device


cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
// memory alloc for x
cudaStat = cudaMalloc (( void **)& d_y ,n* sizeof (*y)); // device
// memory alloc for y
stat = cublasCreate (& handle ); // initialize CUBLAS context
stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x- >d_x
stat = cublasSetVector (n, sizeof (*y) ,y ,1 ,d_y ,1); // cp y- >d_y

float al =2.0; // al =2
// multiply the vector d_x by the scalar al and add to d_y
// d_y = al*d_x + d_y , d_x ,d_y - n- vectors ; al - scalar
stat = cublasSaxpy(handle,n,&al,d_x,1,d_y,1);
stat = cublasGetVector (n, sizeof ( float ) ,d_y ,1 ,y ,1); // cp d_y - >y
printf ("y after Saxpy :\n"); // print y after Saxpy
for(j = 0;j < n;j++) {
  printf (" %2.0f,",y[j]);
}
printf ("\n");
cudaFree (d_x ); // free device memory
cudaFree (d_y ); // free device memory
cublasDestroy ( handle ); // destroy CUBLAS context
free (x); // free host memory
free (y); // free host memory
return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// 2*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
