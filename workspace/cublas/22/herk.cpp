# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string>
#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 
#define RANDOM (rand() % 10000 * 1.00) / 100




# define n 6 // c - nxn matrix
# define k 5 // a - nxk matrix

void PrintMatrix(cuComplex* Matrix, int matriA_row, int matriA_col) {
  int row, col;
  for (row = 0; row < matriA_row; row++) {
    for (col = 0; col < matriA_col; col++) {
      std::cout << Matrix[INDEX(row, col, matriA_row)].x << "+" << Matrix[INDEX(row, col, matriA_row)].y << "*I ";
    }
    std::cout << "\n";
  }
}

int main (int argc, char **argv) {

  int A_row, A_col, C_row, C_col;
  float alpha, beta;
  
  std::cout << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;
  // reading cmd line arguments
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha")))
      alpha = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = atof(argv[loop_count + 1]);
  }
  
  C_row = A_row;
  C_col = A_row;
  
  cudaError_t cudaStatus; // cudaMalloc status
  cublasStatus_t status; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context
  int row, col;
  clock_t clk_start, clk_end;
  
  // data preparation on the host
  cuComplex *HostMatA; // nxk complex matrix a on the host
  cuComplex *HostMatC; // nxn complex matrix c on the host
  HostMatA = new cuComplex[A_row * A_col]; // host memory
  // alloc for a
  HostMatC = new cuComplex[C_row * C_col]; // host memory
  // alloc for c
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix A)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix C)\n");
    return EXIT_FAILURE;
  }
  
  // define the lower triangle of an nxn Hermitian matrix c in
  // lower mode column by column ;
  int ind =11; // c:
  for(col = 0; col < C_col; col++) {           // 11
    for(row = 0; row < C_row; row++) {            // 12 ,17
      if(row >= col) {                                  // 13 ,18 ,22
        HostMatC[INDEX(row, col, C_row)].x = ( float )ind ++;     // 14 ,19 ,23 ,26
        HostMatC[INDEX(row, col, C_row)].y = 0.0 f;                 // 15 ,20 ,24 ,27 ,29
      }                                                           // 16 ,21 ,25 ,28 ,30 ,31 
    }
  }
  // print the lower triangle of c row by row
  std::cout << "lower triangle of C :\n";
  for (row = 0; row < C_row; row++){
    for (col = 0; col < C_col; col++) {
      if(row >= col) {
        std::cout << HostMatC[INDEX(row, col, C_row)].x << "+" << HostMatC[INDEX(row, col, C_row)].y << "*I ";                              
      }
    }
  std::cout << "\n";
  }
  
  //defining a matrix A
  for(col = 0; col < A_col; col++) {           
    for(row = 0; row < A_row; row++) {                      
      HostMatA[INDEX(row, col, A_row)].x = RANDOM;            
      HostMatA[INDEX(row, col, A_row)].y = 0.0f;                   
                   
    }
  }
  // print A row by row
  std::cout << "A:\n";
  PrintMatrix(HostMatA, A_row, A_col);
  
  


  
  // on the device
  cuComplex *DeviceMatA;  // d_a - a on the device
  cuComplex *DeviceMatC;  // d_c - c on the device
  
  cudaStatus = cudaMalloc ((void **)& DeviceMatA , A_row * A_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **)& DeviceMatC, C_row * C_col * sizeof (cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C\n";
    return EXIT_FAILURE;
  }
  // device memory alloc for c
  
  status = cublasCreate (& handle);  // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  } 
  
   // copy matrices from the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA) , HostMatA, A_row, DeviceMatA, A_row); //a -> d_a
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC) , HostMatC, C_row, DeviceMatC, C_row); //c -> d_c
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }
  

  float al =1.0 f; // al =1
  float bet =1.0 f; // bet =1
  // rank -k update of a Hermitian matrix :
  // d_c =al*d_a *d_a ^H +bet *d_c
  // d_c - nxn , Hermitian matrix ; d_a - nxk general matrix ;
  // al ,bet - scalars
  status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  n,k,&al,d a,n,&bet,d c,n);
stat = cublasGetMatrix (n,n, sizeof (*c) ,d_c ,n,c,n); // d_c -> c
printf (" lower triangle of c after Cherk :\n");
for (i=0;i<n;i ++){
for (j=0;j<n;j ++){ // print c after Cherk
if(i >=j)
printf (" %5.0 f +%1.0 f*I",c[ IDX2C (i,j,n)].x,
c[ IDX2C (i,j,n)].y);
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
// lower triangle of c:
// 11+ 0*I
// 12+ 0*I 17+ 0*I
// 13+ 0*I 18+ 0*I 22+ 0*I
// 14+ 0*I 19+ 0*I 23+ 0*I 26+ 0*I
// 15+ 0*I 20+ 0*I 24+ 0*I 27+ 0*I 29+ 0*I
// 16+ 0*I 21+ 0*I 25+ 0*I 28+ 0*I 30+ 0*I 31+ 0*I
// a:
// 11+ 0*I 17+ 0*I 23+ 0*I 29+ 0*I 35+ 0*I
// 12+ 0*I 18+ 0*I 24+ 0*I 30+ 0*I 36+ 0*I
// 13+ 0*I 19+ 0*I 25+ 0*I 31+ 0*I 37+ 0*I
// 14+ 0*I 20+ 0*I 26+ 0*I 32+ 0*I 38+ 0*I
// 15+ 0*I 21+ 0*I 27+ 0*I 33+ 0*I 39+ 0*I
