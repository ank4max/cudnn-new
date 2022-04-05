%%writefile herk.cc
#include <unordered_map>
#include "herk.h"

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

  //! Destroy CuBLAS context
  status  = cublasXtDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Herk<T>::HerkApiCall() {
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

  //! Initializing CUBLAS context
  status = cublasXtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Device Selection
  int devices[1] = { 0 }; 
  status = cublasXtDeviceSelect(handle, 1, devices);
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Set devices fail\n"; 
    return EXIT_FAILURE;   
  }


  /**
   * API call to performs the Hermitian rank-k update : \f$ C = alpha * A * A^H + beta * C \f$
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, k < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  switch (mode) {
    case 'C': {
      float alpha_f = (float)alpha;
      float beta_f = (float)beta;  
      std::cout << "\nCalling XtCherk API\n";
      clk_start = clock();

      status = cublasXtCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, &alpha_f, 
                           (cuComplex *)HostMatrixA, A_row,
                           &beta_f, (cuComplex *)HostMatrixC, C_row); 

        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtCherk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtCherk API call ended\n";
      break;
    }
      
    case 'Z': {
      std::cout << "\nCalling XtZherk API\n";
      clk_start = clock();

      status = cublasXtZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, &alpha,
                           (cuDoubleComplex *)HostMatrixA, A_row,
                           &beta, (cuDoubleComplex *)HostMatrixC, C_row); 
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtZherk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtZherk API call ended\n";
      break;
    }

  }

  std::cout << "\nMatrix C after Xt" << mode << "herk operation is:\n";

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

int mode_C(int A_row, int A_col, int C_row, int C_col, double alpha_real, double beta_real) {
 
  Herk<cuComplex> Cherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, 'C');
  return Cherk.HerkApiCall();

}

int mode_Z(int A_row, int A_col, int C_row, int C_col, double alpha_real, double beta_real) {
  
  Herk<cuDoubleComplex> Zherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, 'Z');
  return Zherk.HerkApiCall();

}

int (*cublas_func_ptr[])(int, int, int, int, double, double) = {
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

  //! Dimension check
  if(A_row <= 0 || A_col <= 0) {
    std::cout << "Minimum Dimension error\n";
    return EXIT_FAILURE;
  }

  C_row = A_row;
  C_col = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, alpha_real, beta_real);
  

  return status;
}
