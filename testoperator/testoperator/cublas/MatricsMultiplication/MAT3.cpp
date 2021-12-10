// Matrix multiplication: C = A * B.
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include<assert.h>
#include <string.h>
#include <math.h>
#include<time.h>



void verify_solution(float *a, float *b, float*n)
{
  
  float temp;
  float epsilon =0.001;
  for(int i =0;i<n;i++)
  {
     for(int j=0;j<n;j++)
     {
        temp=0;
       for(int k=0;k<n;k++)
       {
                temp+=a{k*n+i]* b[j*n+k];
        }
      assert(fabs(c[j*n+1]-temp)<epsilon);
       }
 }
  }
  
 int main()
 {
          int n =1<<10;
         size_t bytes =n*n *sizeof(float);
        float *d_a,*d_b,*d_c;
   
   
         
     float *h_a =(float*)malloc(bytes);
   float  *h_b=(float*)malloc(bytes);
   float *h_c=(float*)malloc(bytes);
   
   
    cudaMalloc(&d_a,bytes);
   cudaMalloc(&d_b,bytes);
   cudaMalloc(&d_c,bytes);
   
   
   
   
   curandGenerator_t prng;
   curandCreateGenerator(&prng,CURAND_RNG_PSEUDO_DEFAULT);
   
   curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long)clock()); 
   
       curandGeneratorUniform(prng,d_a,n*n);
       curandGeneratorUniform(prng,d_b,n*n);
   
   cublasHandle_t handle;
   cublasCreate(&handle);
   
   float alpha =1.0f;
   float beta=0.0f;
   
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha,d_a,n,d_b,n,&beta,d_c,n);
   
   cudaMemcpy(h_a,d_a,bytes,cudaMemcpyDeviceToHost);
   cudaMemcpy(h_b,d_b,bytes,cudaMemcpyDeviceToHost);
   cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);

   verify_solution(h_a,h_b,h_c,n);
   
   printf("completed successfully");
    
   
   
   
   
   
   
  
    
       
