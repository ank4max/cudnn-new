%%writefile ne.cc

#include <unordered_map>
#include "cublas_MatrixTransform_test.h"

template<class T>
MatrixTransform<T>::MatrixTransform(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {
      transformDesc = NULL;
      Adesc = NULL;
      Bdesc = NULL;
      Cdesc = NULL;
      }

template<class T>
void MatrixTransform<T>::FreeMemory() {
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
    
  //! Destroy transform descriptor
  status = cublasLtMatrixTransformDescDestroy(transformDesc);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to destroy transform descriptor \n";
  }
    
  //! Destroy CuBLAS context
  status  = cublasLtDestroy(LtHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int MatrixTransform<T>::MatrixTransformApiCall() {
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
    
  //! Setting up trans for performing matrix-matrix multiplication
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
  status = cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Transform Descriptor creation error\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Transform  descriptor set error for A \n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transb, sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Transform  descriptor set error for B \n";
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

  cudaStream_t stream = 0;
  /**
   * API call to computes the matrix transformation operation on the input matrices A and B, to produce the output matrix C\n 
   * \f$ C = alpha * transform(A) + beta * transform(B) \f$ \n
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS -  If the operation completed successfully. \n
   * CUBLAS_STATUS_NOT_INITIALIZED -  If cuBLASLt handle has not been initialized. \n
   * CUBLAS_STATUS_INVALID_VALUE - If the parameters are in conflict or in an impossible configuration. For example, when A is not NULL, 
        but Adesc is NULL. \n
   * CUBLAS_STATUS_NOT_SUPPORTED - If the current implementation on the selected device doesn't support the configured operation \n
   * CUBLAS_STATUS_ARCH_MISMATCH - If the configured operation cannot be run using the selected device.\n
   * CUBLAS_STATUS_EXECUTION_FAILED - If CUDA reported an execution error from the device. \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Matrix Transform API\n";
      clk_start = clock();

      status = cublasLtMatrixTransform(LtHandle, transformDesc, &alpha, DeviceMatrixA, Adesc, &beta, DeviceMatrixB, Bdesc,
                                       DeviceMatrixC, Cdesc, stream);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Matrix Transform kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Matrix Transform API call ended\n";
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

  std::cout << "\nMatrix C after " << mode << "MatrixTransform operation is:\n";

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

  MatrixTransform<float> MatrixTransform(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S' );
  return MatrixTransform.MatrixTransformApiCall();
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
