%%writefile ne1.cc
#include <unordered_map>
#include "cublas_CherkEx_test.h"

template<class T>
CherkEx<T>::CherkEx(int A_row, int A_col, int C_row, int C_col, float alpha, float beta, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void CherkEx<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostMatrixC)
    delete[] HostMatrixC;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
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
int CherkEx<T>::CherkExApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostMatrixC = new T[C_row * C_col];

  if (!HostMatrixA) {
    std::cout << "!!!! Host memory allocation error (matrixA)\n";
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
   * A is a General Matrix,
   * C is a Hermitian Matrix
   */
  switch (mode) {
    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
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

  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix C from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * API call to performs the Hermitian rank-k update : \f$ C = alpha * A * A^H + beta * C \f$
   * This function is an extension of cublasCherk where the input matrix and output matrix can have a lower precision but the computation
     is still done in the type cuComplex.\n
   * This routine is only supported on GPUs with architecture capabilities equal or greater than 5.0. \n
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, k < 0 \n
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters Atype and Ctype is not supported.\n
   * CUBLAS_STATUS_ARCH_MISMATCH - The device has a compute capabilities lower than 5.0.\n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  switch (mode) {
    case 'C': { 
      std::cout << "\nCalling CherkEx API\n";
      clk_start = clock();

      status = cublasCherkEx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             A_row, A_col, &alpha, (cuComplex *)DeviceMatrixA, 
                             CUDA_C_32F, A_row, &beta, (cuComplex *)DeviceMatrixC, 
                             CUDA_C_32F, C_row); 

        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CherkEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CherkEx API call ended\n";
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

  std::cout << "\nMatrix C after " << mode << "CherkEx operation is:\n";

  //! Print the final resultant Matrix C
  switch (mode) {
    case 'C': {
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col); 
      break;
    }
  }

  long long total_operations = A_row * A_col * C_col;

  //! Print Latency and Throughput of the API
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}
int mode_C(int A_row, int A_col, int C_row, int C_col, float alpha_real, float beta_real) {
 
  CherkEx<cuComplex> CCherkEx(A_row, A_col, C_row, C_col, alpha_real, beta_real, 'C');
  return CCherkEx.CherkExApiCall();

}


int (*cublas_func_ptr[])(int, int, int, int, float, float) = {
  mode_C
};
int main(int argc, char **argv) {
  int A_row, A_col, C_row, C_col, status;
  float alpha_real, beta_real;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['C'] = 0;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  //! Reading cmd line arguments and initializing the parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stof(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Dimension check
  if(A_row <= 0 || A_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  C_row = A_row;
  C_col = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, alpha_real, beta_real);
  

  return status;
}
