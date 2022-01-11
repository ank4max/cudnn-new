#include <iostream>
#include <string>
#include "cublas.h"
#include "cublas_v2.h"
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values 

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(int8_t* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matrix_col; col++) {
      std::cout << unsigned(Matrix[INDEX(row, col, matrix_row)]) << " ";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {
  
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha, beta;

  std::cout << "\n\n" << argv[0] << std::endl;
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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha")))
      alpha = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = atof(argv[loop_count + 1]);
  }
 
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  cudaError_t cudaStatus; 
  cublasStatus_t status; 
  cublasHandle_t handle;

  clock_t clk_start, clk_end;   
 
  int8_t *HostMatA; // mxk matrix A on the host
  int8_t *HostMatB; // kxn matrix B on the host
  int8_t *HostMatC; // mxn matrix C on the host
  
  HostMatA =  new int8_t [A_row * A_col]; // host memory for A
  HostMatB =   new int8_t [B_row * B_col]; // host memory for B
  HostMatC =  new int8_t [C_row * C_col]; // host memory for C
  
   if (HostMatA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }

  int row, col;
  
  
  int ind =11;
  
  
  // define an mxk matrix A column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < A_row; row++) {                                              
    for (col = 0; col < A_col; col++) {                                                   
      HostMatA[INDEX(row, col, A_row)] =   ind++;                                    
    }                                                                                    
  }                                                                               
                      ind=11;                                                         
  // define a kxn matrix B column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < B_row; row++) {                                      
    for (col = 0; col < B_col; col++) {                                                
      HostMatB[INDEX(row, col, B_row)] = ind++;                                          
    }                                                                         
  }                                        
  
   ind=11; 
   // define an mxn matrix C column by column
  // using RANDOM macro to generate random numbers between 0 - 100
  for (row = 0; row < C_row; row++) {                             
    for (col = 0; col < C_col; col++) {                                        
      HostMatC[INDEX(row, col, C_row)] = ind++;                 
    }                                                                  
  }

  
  
  // printing input matrices
  std::cout << "\nMatrix A:\n";
  PrintMatrix(HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:\n";
  PrintMatrix(HostMatB, B_row, B_col);
  std::cout << "\nMatrix C:\n";
  PrintMatrix(HostMatC, C_row, C_col); 
 
  void *workspace;
  size_t workspaceSize = 1024 * 1024 * 8;
  cudaMalloc(&workspace, workspaceSize);
 
  // on the device
  int8_t *DeviceMatA; // d_A - A on the device
  int8_t *DeviceMatB; // d_B - B on the device
  int8_t *DeviceMatC; // d_C - C on the device

  cudaStatus = cudaMalloc ((void **) &DeviceMatA , A_row * A_col * sizeof (*HostMatA));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatB , B_row * B_col * sizeof (*HostMatB));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc ((void **) &DeviceMatC , C_row * C_col * sizeof (*HostMatC));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    return EXIT_FAILURE;   
  }
	
  //copying values	
  cudaMemcpy(DeviceMatA, HostMatA, A_row * A_col, cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceMatB, HostMatB, B_row * B_col, cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceMatC, HostMatC, C_row * C_col, cudaMemcpyHostToDevice);  
  
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
    
  cublasLtHandle_t ltHandle;
  status  = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

  // create matrix descriptors, we are good with the details here so no need to set any extra attributes
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, A_col, A_row, A_col);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, B_row, B_col, B_row);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, C_row, C_col, C_row);

  cublasLtMatmulPreferenceCreate(&preference);
    
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  //cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

  cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
  std::cout<<"List of algos :"<<returnedResults<<std::endl;
	
  clk_start = clock();  
        
  status = cublasLtMatmul(ltHandle,
        operationDesc,
        &alpha,
        DeviceMatA,
        Adesc,
        DeviceMatB,
        Bdesc,
        &beta,
        DeviceMatC,
        Cdesc,
        DeviceMatC,
        Cdesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();

  cudaMemcpy(HostMatC, DeviceMatC, C_row * C_col, cudaMemcpyDeviceToHost);
  std::cout << "\nMatriz C after lTmatmul operation is:\n";
  PrintMatrix(HostMatC, C_row, C_col);
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " <<THROUGHPUT(clk_start, clk_end) << "\n\n";

  
  cudaStatus = cudaFree (DeviceMatA); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatB); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B" << std::endl;
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaFree (DeviceMatC); // free device memory
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
    return EXIT_FAILURE;   
  }
  
  status  = cublasLtDestroy (ltHandle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  }

 



delete []HostMatA;
delete []HostMatB;
delete []HostMatC;

return 0;
}
