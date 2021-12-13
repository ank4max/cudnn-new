#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
//printf("\n %f",P[1]);
int i,j;
for(i=0;i<uHP;i++){

    printf("\n");

    for(j=0;j<uWP;j++)
        printf("%f ",P[index(i,j,uHP)]);
        //printf("%f ",P[i*uWP+j]);
}
}

int  main (int argc, char** argv) {
    cublasStatus status;
    cublasHandle_t handle ;
    
    
     float *A = (float*)malloc(HA*WA*sizeof(float));
        float *B = (float*)malloc(HB*WB*sizeof(float));
        float *C = (float*)malloc(HC*WC*sizeof(float));
    if (A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (C == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
        }
        
        status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;


   for (i=0;i<HA;i++)
     for (j=0;j<WA;j++)
        A[index(i,j,HA)] = (float) index(i,j,HA);   
        for (i=0;i<HB;i++)
     for (j=0;j<WB;j++)
        B[index(i,j,HB)] = (float) index(i,j,HB); 
    /*
    for (i=0;i<HA*WA;i++)
    A[i]=(float) i;
    for (i=0;i<HB*WB;i++)
    B[i]=(float) i;         */  


        float* AA; float* BB; float* CC; 
        //using cuda malloc
        
        cudaStat = cudaMalloc ((void**)&AA, HA*WA*sizeof(*A));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
        
        cudaStat = cudaMalloc ((void**)&BB, HB*WB*sizeof(*B));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
        
        
        cudaStat = cudaMalloc ((void**)&CC, HC*WC*sizeof(*C));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
        
        
        *SET MATRIX*/
        status=cublasSetMatrix(HA,WA,sizeof(float),A,HA,AA,HA);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }

        status=cublasSetMatrix(HB,WB,sizeof(float),B,HB,BB,HB);
        if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
        }
        
        
        
        
        
        
        

