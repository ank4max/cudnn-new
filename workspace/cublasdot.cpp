
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <time.h>
# include <iostream>

int main ( int argc,char **argv ) {

  //initializing size of vector with command line arguement
  cudaError_t cudaStat ; 
  cublasStatus_t stat ; 
  cublasHandle_t handle ;
  clock_t start, end;
  int n= atoi(argv[1]);
  int j; 
  
  //pointers x and y pointing  to vectors
  float * x;             
  float * y; 
  
  //host memory allocation for vectors
  x = ( float *) malloc (n* sizeof (*x)); 
  y = ( float *) malloc (n* sizeof (*y)); 

  //setting up values in x and y vectors
  for(j = 0;j < n; j++) {
    x[j] = ( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}
  }

  for (j = 0; j < n; j++) {
    y[j] = ( float )j; 
  }
  
  //printing the initial values in vector x which is same as y
  printf ("x,y:\n");
  for (j = 0; j < n; j++) {
    printf (" %2.0f,",x[j]); 
  }
  printf ("\n");
  
  // Pointers for device memory allocation
  float * d_x; 
  float * d_y; 
  cudaStat = cudaMalloc (( void **)& d_x, n* sizeof (*x)); 
  cudaStat = cudaMalloc (( void **)& d_y, n* sizeof (*y)); 
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  stat = cublasCreate (& handle );
  stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); 
  stat = cublasSetVector (n, sizeof (*y) ,y ,1 ,d_y ,1); 

  float result ;
  // performing dot product operation and storing result in result variable
  start=clock();
  stat=cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
  end=clock();
  
  //printing the final result
  printf ("dot product x.y:\n");
  printf (" %7.0f",result ); 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  
  //freeing device memory
  cudaFree (d_x ); 
  cudaFree (d_y );
  
  //destroying cublas context and freeing host memory
  cublasDestroy ( handle ); 
  free (x); 
  free (y); 
  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// dot product x.y: // x.y=
// 55 // 1*1+2*2+3*3+4*4+5*5
