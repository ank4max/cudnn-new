#include <unordered_map>
#include "cublas_matmul_test.h"

template<class T>
Matmul<T>::Matmul(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Matmul<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostMatrixB)
    delete[] HostMatrixB;

  if (HostMatrixC)
    delete[] HostMatrixC;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixB);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixC);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasLtDestroy(LtHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }

  status = cublasLtMatmulDescDestroy(operationDesc);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to destroy operation descriptor \n";
  }

  status = cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to destroy preference \n";
  }
}

template<class T>
int Matmul<T>::MatmulApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostMatrixB = new T[B_row * B_col];
  HostMatrixC = new T[C_row * C_col];

  if (!HostMatrixA) {
    std::cout << "!!!! Host memory allocation error (matrixA)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostMatrixB) {
    std::cout << "!!!! Host memory allocation error (matrixB)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostMatrixC) {
    std::cout << "!!!! Host memory allocation error (matrixC)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrices based on mode passed,
   * A, B and C are general matrices
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<float>((float *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
          
      break;
    }
  }
  
  //! Allocating Device Memory for Matrices using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceMatrixA, A_row * A_col * sizeof(*HostMatrixA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceMatrixB, B_row * B_col * sizeof(*HostMatrixB));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceMatrixC, C_row * C_col * sizeof(*HostMatrixC));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Initializing CUBLAS context
  status = cublasLtCreate(&LtHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Setting workspace size  
  void *workspace = NULL;
  size_t workspaceSize = 0 ;
    
  //! Setting up trans for performing matrix-matrix multiplication
  int returnedResults = 0;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  
  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix A from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix B from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix C from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Creating necessary descriptors
  status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Operation Descriptor creation error\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Descriptor set attribute error for A\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Descriptor set attribute error for B\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, A_row, A_col, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Layout create error for A \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, B_row, B_col, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Layout create error for B \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, C_row, C_col, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Layout create error for C \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Preference creation error \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Preference Set Attribute API error \n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  int requestedAlgoCount = 1;
  cudaStream_t stream = 0;
  //! Getting algorithm for Matmul operation
  status = cublasLtMatmulAlgoGetHeuristic(LtHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, requestedAlgoCount, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "MatmulAlgoGetHeuristic API error \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  
  /**
   * API call to performs matrix - matrix multiplication : \f$ C = alpha * A * B + beta * C \f$ \n
   * This function supports both in-place matrix multiplication (C == D and Cdesc == Ddesc) and out-of-place matrix multiplication 
       (C != D, both matrices must have the same data type, number of rows, number of columns, batch size, and memory order). 
   * In the out-of-place case, the leading dimension of C can be different from the leading dimension of D. Specifically the leading  
       dimension of C can be 0 to achieve row or column broadcast. If Cdesc is omitted, this function assumes it to be equal to Ddesc.
   * Using a regular data ordering:
   * All matrix pointers must be 4-byte aligned. For even better performance, this condition should hold with 16 instead of 4.
   * Leading dimensions of matrices A, B, C must be multiplies of 4.
   * Only "TN" format is supported - A must be transposed and B non-transposed.
   * Dimensions m and k must be multiplies of 4.
   * Using the IMMA-specific data ordering - CUBLASLT_ORDER_COL32 for matrices A,C,D, and CUBLASLT_ORDER_COL4_4R2_8C (on Turing or Ampere 
       architecture) or CUBLASLT_ORDER_COL32_2R_4R4 (on Ampere architecture) for matrix B:
   * Leading dimensions of matrices A, B, C must fulfill conditions specific to the memory ordering (see cublasLtOrder_t).
   * Matmul descriptor must specify CUBLAS_OP_T on matrix B and CUBLAS_OP_N (default) on matrix A and C.
   * If scaleType CUDA_R_32I is used, the only supported values for alpha and beta are 0 or 1.
   * When using regular memory order and when compute type is 32I, input type is R_8I and output type is R_8I, only "TN" format is  
       supported - "A" must be transposed and "B" non-transposed.
   * IMMA kernel with computeType=32I and Ctype=CUDA_R_8I supports per row scaling (see CUBLASLT_POINTER_MODE_DEVICE_VECTOR and 
       CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO in cublasLtPointerMode_t) as well as ReLU and Bias epilogue modes (see 
       CUBLASLT_MATMUL_DESC_EPILOGUE in cublasLtMatmulDescAttributes_t).
   * These can only be used with planar layout (CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET != 0).
   * ReLU, dReLu, GELU, dGELU and Bias epilogue modes (see CUBLASLT_MATMUL_DESC_EPILOGUE in cublasLtMatmulDescAttributes_t) are not 
       supported when D matrix memory order is defined as CUBLASLT_ORDER_ROW. For best performance when using the bias vector, specify beta == 0 and CUBLASLT_POINTER_MODE_HOST.
   * Use of CUBLAS_ORDER_ROW together with CUBLAS_OP_C (hermitian operator) is not supported unless all of A, B, C, D matrices are defined 
       with CUBLAS_ORDER_ROW.
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS -  If the operation completed successfully. \n
   * CUBLAS_STATUS_NOT_INITIALIZED -  If cuBLASLt handle has not been initialized. \n
   * CUBLAS_STATUS_INVALID_VALUE - If the parameters are unexpectedly NULL, in conflict or in an impossible configuration. For example, 
        when workspaceSizeInBytes is less than workspace required by the configured algo. \n
   * CUBLAS_STATUS_NOT_SUPPORTED - If the current implementation on the selected device doesn't support the configured operation \n
   * CUBLAS_STATUS_ARCH_MISMATCH - If the configured operation cannot be run using the selected device.\n
   * CUBLAS_STATUS_EXECUTION_FAILED - If CUDA reported an execution error from the device. \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Matmul API\n";
      clk_start = clock();

      status = cublasLtMatmul(LtHandle, operationDesc, &alpha, DeviceMatrixA, Adesc, DeviceMatrixB, Bdesc, &beta,
                              DeviceMatrixC, Cdesc, DeviceMatrixC, Cdesc, &heuristicResult.algo, workspace, workspaceSize, stream);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Matmul kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Matmul API call ended\n";
      break;
    }
    
  }
  
  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                           DeviceMatrixC, C_row, HostMatrixC, C_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output matrix C from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix C after " << mode << "Matmul operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      break;
    }
  }

  long long total_operations = A_row * A_col * B_col;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, float alpha, float beta) {


  Matmul<float> matmul(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S' );
  return matmul.MatmulApiCall();
}


int (*cublas_func_ptr[])(int, int, int, int, int, int, float, float) = {
  mode_S
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, status;
  float alpha, beta;
  char mode;
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha")))
      alpha = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Dimension check
  if (A_row <= 0 || A_col <= 0 || B_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }
  
  //! initializing values for matrix B and C
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, alpha, 
                                                beta);
  
  return status;
}
