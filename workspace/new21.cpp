# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <time.h>
# include <iostream>

char* Substr(char* cInputArr, int nBegin, int nLen)
{
    char* pcResStr = new char[nLen + 1];
    for (int i = 0; i < nLen; i++)
        pcResStr[i] = *(cInputArr + nBegin + i);
    pcResStr[nLen] = 0;
    return pcResStr;
}


int main ( int argc,char **argv ) {

  //initializing size of vector with command line arguement
  cudaError_t cudaStat ; 
  cublasStatus_t stat ; 
  cublasHandle_t handle ;
  clock_t start, end;
  int nLenA, nLenB;
  int nLenVector;
  
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
    
  for (int i = 1; i < 3; i++) {
    if (!strcmp(Substr(argv[i], 1, 4), "lenA"))
      nLenA = atoi(argv[i] + 5);
    else if (!strcmp(Substr(argv[i], 1, 4), "lenB"))
      nLenB = atoi(argv[i] + 5);
  }

  if(nLenA != nLenB) {
    return EXIT_FAILURE ;
  }
  else
  {
    nLenVector = nLenA;
  }
    
  int j; 
  
  //pointers x and y pointing  to vectors
  float * pfHostVecX;             
  float * pfHostVecY; 
  
  //host memory allocation for vectors
  pfHostVecX = ( float *) malloc ( nLenVector* sizeof (*pfHostVecX)); 
  pfHostVecY = ( float *) malloc ( nLenVector* sizeof (*pfHostVecY)); 
  
  if (pfHostVecX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (vector x )\n");
    return EXIT_FAILURE;
  }
   
  if (pfHostVecY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (vector y )\n");
    return EXIT_FAILURE;
  }
  


  //setting up values in x and y vectors
  for(j = 0;j < nLenVector; j++) {
    pfHostVecX[j] = ( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}
  }

  for (j = 0; j < nLenVector ; j++) {
    pfHostVecY[j] = ( float )j; 
  }
  
  //printing the initial values in vector x and vector y
  printf ("x:\n");
  for (j = 0; j < nLenVector; j++) {
    printf (" %2.0f,",pfHostVecX[j]); 
  }
  printf ("\n");
  
   printf ("y:\n");
  for (j = 0; j < nLenVector; j++) {
    printf (" %2.0f,",pfHostVecY[j]); 
  }
  printf ("\n");
  
  // Pointers for device memory allocation
  float * pfDevVecX; 
  float * pfDevVecY; 
  
  cudaStat = cudaMalloc (( void **)& pfDevVecX, nLenVector* sizeof (*pfHostVecX));
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;
  }
  
  cudaStat = cudaMalloc (( void **)& pfDevVecY, nLenVector* sizeof (*pfHostVecY));
  
  if( cudaStat != cudaSuccess) {
    printf(" the device memory allocation failed\n");
    return EXIT_FAILURE;   
  }
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  stat = cublasCreate (& handle );
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  stat = cublasSetVector (nLenVector, sizeof (*pfHostVecX) , pfHostVecX, 1, pfDevVecX, 1); 
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for X on gpu\n");
    return EXIT_FAILURE;
  }
  
  stat = cublasSetVector (nLenVector, sizeof (*pfHostVecY), pfHostVecY, 1, pfDevVecY, 1); 
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for Y on gpu\n");
    return EXIT_FAILURE;
  }

  float fResult ;
  // performing dot product operation and storing result in result variable
  start=clock();
  stat=cublasSdot(handle, nLenVector, pfDevVecX, 1, pfDevVecY, 1, &fResult);
  end=clock();
  
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  //printing the final result
  printf ("dot product x.y:\n");
  printf (" %7.0f",fResult ); 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end - start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  
  //freeing device memory
  cudaStat = cudaFree (pfDevVecX );
  if( cudaStat != cudaSuccess) {
    printf(" memory free error on device for vector x\n");
    return EXIT_FAILURE;
  }
  
  cudaFree (pfDevVecY );
  if( cudaStat != cudaSuccess) {
    printf(" memory free error on device for vector y\n");
    return EXIT_FAILURE;
  }
  
  //destroying cublas context and freeing host memory
  cublasDestroy ( handle ); 
  free (pfHostVecX); 
  free (pfHostVecY); 
  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// dot product x.y: // x.y=
// 55 // 1*1+2*2+3*3+4*4+5*5
