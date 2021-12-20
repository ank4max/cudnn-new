# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <time.h>
# include <iostream>

char* substr(char* arr, int begin, int len)
{
    char* res = new char[len + 1];
    for (int i = 0; i < len; i++)
        res[i] = *(arr + begin + i);
    res[len] = 0;
    return res;
}


int main ( int argc,char **argv ) {

  //initializing size of vector with command line arguement
  cudaError_t cudaStat ; 
  cublasStatus_t stat ; 
  cublasHandle_t handle ;
  clock_t start, end;
  int lenA, lenB;
  
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  
  for (int i = 1; i < 3; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(substr(argv[i], 1, 4), "lenA"))
      lenA = atoi(argv[i] + 5);
    else if (!strcmp(substr(argv[i], 1, 4), "lenB"))
      lenB = atoi(argv[i] + 5);

  if(lenA != lenB) {
    return EXIT_FALURE
  }
  else
  {
    n = lenA;
  }
    
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
  
  //printing the initial values in vector x and vector y
  printf ("x:\n");
  for (j = 0; j < n; j++) {
    printf (" %2.0f,",x[j]); 
  }
  printf ("\n");
  
   printf ("y:\n");
  for (j = 0; j < n; j++) {
    printf (" %2.0f,",y[j]); 
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
