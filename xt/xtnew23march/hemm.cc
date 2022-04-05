%%writefile hemm.cc
#include "hemm.h"
#include <unordered_map>
#include <bits/stdc++.h>

template<class T>
Hemm<T>::Hemm(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Hemm<T>::FreeMemory(){
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;
  
  if (HostMatrixB)
    delete[] HostMatrixB;

  if (HostMatrixC)
    delete[] HostMatrixC;

  //! Destroy CuBLAS context
  status  = cublasXtDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Hemm<T>::HemmApiCall() {
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
   * Switch Case - To Initialize and Print input matrices based on mode passed
   * A is a Hermitian matrix, B and C are general matrices
   */
  switch (mode) {
    case 'C': {
      util::InitializeSymmetricComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      break; 
    }
                        
    case 'Z': {
      util::InitializeSymmetricComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
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
   * API call to performs matrix - matrix multiplication : \f$ C = alpha * A * B + beta * C \f$
   */
     
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  switch (mode) {
    case 'C': {
      std::cout << "\nCalling XtChemm API\n";
      clk_start = clock();

      status = cublasXtChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            A_row, B_col, (cuComplex *)&alpha,
                            (cuComplex *)HostMatrixA, A_row, 
                            (cuComplex *)HostMatrixB, B_row, (cuComplex *)&beta, 
                            (cuComplex *)HostMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtChemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtChemm API call ended\n";
      break;
    }
  
    case 'Z': {
      std::cout << "\nCalling XtZhemm API\n";
      clk_start = clock();

      status = cublasXtZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            A_row, B_col, (cuDoubleComplex *)&alpha,
                            (cuDoubleComplex *)HostMatrixA, A_row, 
                            (cuDoubleComplex *)HostMatrixB, B_row, 
                            (cuDoubleComplex *)&beta, (cuDoubleComplex *)HostMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtZhemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtZhemm API call ended\n";
      break;
    }
  }
  
  std::cout << "\nMatriz C after Xt" << mode << "hemm operation is:\n";

  switch (mode) {
    case 'C': {
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }

    case 'Z': {
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }
  }

  long long total_operations = A_row * A_col * B_col;  
  
  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";
  
  FreeMemory();
  return EXIT_SUCCESS;      
}

int mode_C(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Hemm<cuComplex> Chemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'C');
  return Chemm.HemmApiCall();
 
}

int mode_Z(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Hemm<cuDoubleComplex> Zhemm(A_row, A_col,B_row, B_col, C_row, C_col, alpha, beta, 'Z');
  return Zhemm.HemmApiCall(); 
}


int (*cublas_func_ptr[])(size_t, size_t, size_t, size_t, size_t, size_t, double, double, double, double) = {
   mode_C, mode_Z
};


int main(int argc, char **argv) {
  size_t A_row, A_col, B_row, B_col, C_row, C_col;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
  char *end;
  
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

  //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    if (!(cmd_argument.compare("-A_row")))
      A_row = std::strtoull((argv[loop_count + 1]), &end, 10);
      
    else if (!(cmd_argument.compare("-B_column")))
      B_col = std::strtoull((argv[loop_count + 1]), &end, 10);

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
  }
	
  //! Dimension check	
  if (A_row <= 0 || B_col <= 0) {
    std::cout << "Minimum Dimension error\n";
    return EXIT_FAILURE;
  }	
 
  //! Initializing values for matrix B and C
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, 
	              	                        alpha_real, alpha_imaginary, beta_real, beta_imaginary);

  return status;
}
