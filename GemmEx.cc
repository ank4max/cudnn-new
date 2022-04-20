%%writefile GemmEx.cc
#include <unordered_map>
#include "GemmEx.h"

template<class T>
GemmEx<T>::GemmEx(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode, char algo)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode), algo(algo) {}

template<class T>
void GemmEx<T>::FreeMemory() {
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
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GemmEx<T>::GemmExApiCall() {
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
  
  
    if (mode == 'S' || mode == 'C')  {
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<float>((float *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
          
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
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
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

  //! Algorithm Selection
  if (algo == 'S') {
    cublas_algo = CUBLAS_GEMM_DEFAULT;
    std::cout <<"\nUsing CUBLAS_GEMM_DEFAULT\n";
  }

 /* 
  else if (algo == 'D') {
    cublas_algo = CUBLAS_GEMM_ALGO0 to CUBLAS_GEMM_ALGO23;
    std::cout << "\nusing CUBLAS_GEMM_ALGO0 to CUBLAS_GEMM_ALGO23\n";
  }
  else if (algo == 'C') {
    cublas_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    std::cout << "\nCUBLAS_GEMM_DEFAULT_TENSOR_OP\n";
  }
  else if (algo == 'Z') {
    cublas_algo = CUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO15_TENSOR_OP;
    std::cout <<"\nCUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO15_TENSOR_OP\n"; 
  }
  */
  
  /**
   * API call to performs matrix - matrix multiplication : \f$ C = alpha * A * B + beta * C \f$
   * This function is an extension of cublas<t>gemm that allows the user to individually specify the data types for each of the A, B and C matrices.
   * The precision of computation and the GEMM algorithm to be run
   * The second variant of cublasGemmEx function is provided for backward compatibility with C++ applications code, where the computeType parameter is of cudaDataType instead of cublasComputeType_t. 
   * C applications would still compile with the updated function signature.
   * CUBLAS_COMPUTE_32I and CUBLAS_COMPUTE_32I_PEDANTIC compute types are only supported with A, B being 4-byte aligned and lda, ldb being multiples of 4.
   * cublasGemmEx routine is run for the algorithms in the following table. Note: for NVIDIA Ampere Architecture GPUs and beyond, i.e. SM version >= 80, 
   * The algorithms below are equivalent to CUBLAS_GEMM_DEFAULT or CUBLAS_GEMM_DEFAULT_TENSOR_OP respectively.
   * Specifying algorithm >= 99 for a single precision operation is equivalent to using CUBLAS_COMPUTE_32F_FAST_16F compute type, 
   * even if math mode or compute type are specified to be CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_FAST_TF32.
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_ARCH_MISMATCH - cublasGemmEx is only supported for GPU with architecture capabilities equal or greater than 5.0
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters Atype, Btype and Ctype or the algorithm, algo is not supported
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n, k < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling GemmEx API cublas compute data type\n";
      clk_start = clock();

      status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (float *)&alpha,
                           (float *)DeviceMatrixA, CUDA_R_32F, A_row,
                           (float *)DeviceMatrixB, CUDA_R_32F, B_row, (float *)&beta,
                           (float *)DeviceMatrixC, CUDA_R_32F, C_row, CUBLAS_COMPUTE_32F, cublas_algo);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  GemmEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "GemmEx API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling GemmEx API with cuda compute data type\n";
      clk_start = clock();

      status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (float *)&alpha,
                           (float *)DeviceMatrixA, CUDA_R_32F, A_row,
                           (float *)DeviceMatrixB, CUDA_R_32F, B_row, (float *)&beta,
                           (float *)DeviceMatrixC, CUDA_R_32F, C_row, CUDA_R_32F, cublas_algo);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  GemmEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "GemmEx API call ended\n";
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

  std::cout << "\nMatrix C after " << "GemmEx operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'C': {
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

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary, char algo) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  GemmEx<float> SgemmEx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S', algo);
  return SgemmEx.GemmExApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary, char algo) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  GemmEx<float> SgemmEx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S', algo);
  return SgemmEx.GemmExApiCall();

}


int (*cublas_func_ptr[])(int, int, int, int, int, int, double, double, double, double, char) = {
  mode_S, mode_C
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
  char algo;
  
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['C'] = 1;


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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-algo")))
      algo = *(argv[loop_count + 1]);
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

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col,   C_row, C_col, alpha_real, 
                                                alpha_imaginary, beta_real,   beta_imaginary, algo);
  
  return status;
}
