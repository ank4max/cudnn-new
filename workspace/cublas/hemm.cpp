# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# define IDX2C (i ,j , ld ) ((( j )*( ld ))+( i ))

# define m 6 // a - mxm matrix
# define n 5 // b,c - mxn matrices
int main ( void ){
  
  cudaError_t cudaStat ; // cudaMalloc status
  cublasStatus_t stat ; // CUBLAS functions status
  cublasHandle_t handle ; // CUBLAS context
int i,j; // i-row index , j-col. ind.
// data preparation on the host
cuComplex *a; // mxm complex matrix a on the host
cuComplex *b; // mxn complex matrix b on the host
cuComplex *c; // mxn complex matrix c on the host
a=( cuComplex *) malloc (m*m* sizeof ( cuComplex )); // host memory
// alloc for a
b=( cuComplex *) malloc (m*n* sizeof ( cuComplex )); // host memory
// alloc for b
c=( cuComplex *) malloc (m*n* sizeof ( cuComplex )); // host memory
// alloc for c
// define the lower triangle of an mxm Hermitian matrix a in
// lower mode column by column
int ind =11; // a:
for(j=0;j<m;j ++){ // 11
for(i=0;i<m;i ++){ // 12 ,17
if(i >=j){ // 13 ,18 ,22
a[ IDX2C (i,j,m)].x=( float )ind ++; // 14 ,19 ,23 ,26
a[ IDX2C (i,j,m)].y =0.0 f; // 15 ,20 ,24 ,27 ,29
} // 16 ,21 ,25 ,28 ,30 ,31
}
}
// print the lower triangle of a row by row
printf (" lower triangle of a:\n");
for (i=0;i<m;i ++){
for (j=0;j<m;j ++){
if(i >=j)
printf (" %5.0 f +%2.0 f*I",a[ IDX2C (i,j,m)].x,
a[ IDX2C (i,j,m)].y);
}
printf ("\n");
}
// define mxn matrices b,c column by column
ind =11; // b,c:
for(j=0;j<n;j ++){ // 11 ,17 ,23 ,29 ,35
for(i=0;i<m;i ++){ // 12 ,18 ,24 ,30 ,36
b[ IDX2C (i,j,m)].x=( float )ind; // 13 ,19 ,25 ,31 ,37
b[ IDX2C (i,j,m)].y =0.0 f; // 14 ,20 ,26 ,32 ,38
c[ IDX2C (i,j,m)].x=( float )ind; // 15 ,21 ,27 ,33 ,39
c[ IDX2C (i,j,m)].y =0.0 f; // 16 ,22 ,28 ,34 ,40
ind ++;
}
}
// print b(=c) row by row
printf ("b,c:\n");
for (i=0;i<m;i ++){
for (j=0;j<n;j ++){
printf (" %5.0 f +%2.0 f*I",b[ IDX2C (i,j,m)].x,
b[ IDX2C (i,j,m)].y);
}
printf ("\n");
}
// on the device
cuComplex * d_a; // d_a - a on the device
cuComplex * d_b; // d_b - b on the device
cuComplex * d_c; // d_c - c on the device
cudaStat = cudaMalloc (( void **)& d_a ,m*m* sizeof ( cuComplex ));
// device memory alloc for a
cudaStat = cudaMalloc (( void **)& d_b ,n*m* sizeof ( cuComplex ));
// device memory alloc for b
cudaStat = cudaMalloc (( void **)& d_c ,n*m* sizeof ( cuComplex ));
// device memory alloc for c
stat = cublasCreate (& handle ); // initialize CUBLAS context
// copy matrices from the host to the device
stat = cublasSetMatrix (m,m, sizeof (*a) ,a,m,d_a ,m); //a -> d_a
stat = cublasSetMatrix (m,n, sizeof (*b) ,b,m,d_b ,m); //b -> d_b
stat = cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); //c -> d_c
cuComplex al ={1.0f ,0.0 f}; // al =1
cuComplex bet ={1.0f ,0.0 f}; // bet =1
// Hermitian matrix - matrix multiplication :
// d_c =al*d_a *d_b +bet *d_c ;
// d_a - mxm hermitian matrix ; d_b ,d_c - mxn - general matices ;
// al ,bet - scalars
stat=cublasChemm(handle,CUBLAS SIDE LEFT,CUBLAS FILL MODE LOWER,
m,n,&al,d a,m,d b,m,&bet,d c,m);
stat = cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); // d_c -> c
printf ("c after Chemm :\n");
for (i=0;i<m;i ++){
for (j=0;j<n;j ++){ // print c after Chemm
printf (" %5.0 f +%1.0 f*I",c[ IDX2C (i,j,m)].x,
c[ IDX2C (i,j,m)].y);
}
printf ("\n");
}
cudaFree (d_a ); // free device memory


cudaFree (d_b ); // free device memory
cudaFree (d_c ); // free device memory
cublasDestroy ( handle ); // destroy CUBLAS context
free (a); // free host memory
free (b); // free host memory
free (c); // free host memory
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
