#include <unordered_map>
#include "cublas_herk_test.h"

template<class T>
Herk<T>::Herk(int A_row, int A_col, int C_row, int C_col, double alpha, double beta, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Herk<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Herk<T>::HerkApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostMatrixC = new T[C_row * C_col];

  if (!HostMatrixA) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  if (!HostMatrixC) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
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
                            
    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
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
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * API call to performs the Hermitian rank- k update : C = alpha * A * A^H + beta * C
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below :
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, k <0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU
   */
  switch (mode) {
    case 'C': {
      float alpha_f = (float)alpha;
      float beta_f = (float)beta;  
      std::cout << "\nCalling Cherk API\n";
      clk_start = clock();

      status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, &alpha_f, 
                           (cuComplex *)DeviceMatrixA, A_row,
                           &beta_f, (cuComplex *)DeviceMatrixC, C_row); 

        
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cherk kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cherk API call ended\n";
      break;
    }
      
    case 'Z': {
      std::cout << "\nCalling Zherk API\n";
      clk_start = clock();

      status = cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, &alpha,
                           (cuDoubleComplex *)DeviceMatrixA, A_row,
                           &beta, (cuDoubleComplex *)DeviceMatrixC, C_row); 
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zherk kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zherk API call ended\n";
      break;
    }

  }

  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()

  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                           DeviceMatrixC, C_row, HostMatrixC, C_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix C after " << mode << "herk operation is:\n";

  //! Print the final resultant Matrix C
  switch (mode) {
    case 'C': {
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col); 
      break;
    }

    case 'Z': {
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col); 
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

void mode_C(int A_row, int A_col, int C_row, int C_col, double alpha_real, double beta_real) {
 
  Herk<cuComplex> Cherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, 'C');
  Cherk.HerkApiCall();

}

void mode_Z(int A_row, int A_col, int C_row, int C_col, double alpha_real, double beta_real) {
  
  Herk<cuDoubleComplex> Zherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, 'Z');
  Zherk.HerkApiCall();

}


void (*cublas_func_ptr[])(int, int, int, int, double, double) = {
  mode_C, mode_Z
};
int main(int argc, char **argv) {
  int A_row, A_col, C_row, C_col, status;
  double alpha_real, beta_real;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['C'] = 0;
  mode_index['Z'] = 1;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // Reading cmd line arguments and initializing the parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  C_row = A_row;
  C_col = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, alpha_real, beta_real);
  

  return EXIT_SUCCESS;
}
