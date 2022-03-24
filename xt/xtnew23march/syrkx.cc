#include <unordered_map>
#include "syrkx.h"
#include <bits/stdc++.h>


template<class T>
Syrkx<T>::Syrkx(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Syrkx<T>::FreeMemory() {
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
int Syrkx<T>::SyrkxApiCall() {
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
   * A and B are general Matrices,
   * C is a symmetric Matrix
   */
  switch (mode) {
    case 'S': {
      util::InitializeMatrixXt<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrixXt<float>((float *)HostMatrixB, B_row, B_col);
      util::InitializeSymmetricMatrixXt<float>((float *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricMatrixXt<float>((float *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintMatrixXt<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrixXt<float>((float *)HostMatrixB, B_row, B_col);
          
      break;
    }

    case 'D': {
      util::InitializeMatrixXt<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrixXt<double>((double *)HostMatrixB, B_row, B_col);
      util::InitializeSymmetricMatrixXt<double>((double *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricMatrixXt<double>((double *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintMatrixXt<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrixXt<double>((double *)HostMatrixB, B_row, B_col); 
      break;  
    }

    case 'C': {
      util::InitializeComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeSymmetricComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      break; 
    }
                            
    case 'Z': {
      util::InitializeComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      util::InitializeSymmetricComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
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
    std::cout << status; 
    return EXIT_FAILURE;   
  }
  
  
  /**
   * API call to performs a variation of the symmetric rank- k update : \f$ C = alpha * A * B ^ T  + beta * C \f$
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, k < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling XtSsyrkx API\n";
      clk_start = clock();

      status = cublasXtSsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            A_row, A_col, (float *)&alpha, 
                            (float *)HostMatrixA, A_row, (float *)HostMatrixB, 
                            B_row, (float *)&beta, (float *)HostMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtSsyrkx kernel execution error\n";
        std::cout<<status;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtSsyrkx API call ended\n";
      break;
    }
                            
    case 'D': {
      std::cout << "\nCalling XtDsyrkx API\n";
      clk_start = clock();

      status = cublasXtDsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            A_row, A_col, (double *)&alpha,
                            (double *)HostMatrixA, A_row, (double *)HostMatrixB, 
                            B_row, (double *)&beta, (double *)HostMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtDsyrkx kernel execution error\n";
        std::cout<<status;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtDsyrkx API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling XtCsyrkx API\n";
      clk_start = clock();

      status = cublasXtCsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            A_row, A_col, (cuComplex *)&alpha,
                            (cuComplex *)HostMatrixA, A_row, 
                            (cuComplex *)HostMatrixB, B_row,
                            (cuComplex *)&beta, (cuComplex *)HostMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtCsyrkx kernel execution error\n";
        std::cout<<status;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtCsyrkx API call ended\n";
      break;
    }
      
    case 'Z': {
      std::cout << "\nCalling XtZsyrkx API\n";
      clk_start = clock();

      status = cublasXtZsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            A_row, A_col, (cuDoubleComplex *)&alpha,
                            (cuDoubleComplex *)HostMatrixA, A_row,
                            (cuDoubleComplex *)HostMatrixB, B_row,
                            (cuDoubleComplex *)&beta,
                            (cuDoubleComplex *)HostMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtZsyrkx kernel execution error\n";
        std::cout<<status;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtZsyrkx API call ended\n";
      break;
    }

  }

  std::cout << "\nMatrix C after Xt" << mode << "syrkx operation is:\n";

  //! Print the final resultant Matrix C
  switch (mode) {
    case 'S': {
      util::PrintSymmetricMatrixXt<float>((float *)HostMatrixC, C_row, C_col); 
      break;
    }

    case 'D': {
      util::PrintSymmetricMatrixXt<double>((double *)HostMatrixC, C_row, C_col);  
      break;
    }

    case 'C': {
      util::PrintSymmetricComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }

    case 'Z': {
      util::PrintSymmetricComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }

  }

  long long total_operations = A_row * A_col * B_col;

  //! Print Latency and Throughput of the API
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Syrkx<float> SSyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S');
  SSyrkx.SyrkxApiCall();
}

void mode_D(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  double alpha = alpha_real;
  double beta = beta_real;

  Syrkx<double> DSyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'D');
  DSyrkx.SyrkxApiCall();
}

void mode_C(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Syrkx<cuComplex> CSyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'C');
  CSyrkx.SyrkxApiCall();
}

void mode_Z(size_t A_row, size_t A_col, size_t B_row, size_t B_col, size_t C_row, size_t C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Syrkx<cuDoubleComplex> ZSyrkx(A_row, A_col,B_row, B_col, C_row, C_col, alpha, beta, 'Z');
  ZSyrkx.SyrkxApiCall();

}


void (*cublas_func_ptr[])(size_t, size_t, size_t, size_t, size_t, size_t, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};


int main(int argc, char **argv) {
  size_t A_row, A_col, B_row, B_col, C_row, C_col;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
  char *end;
  
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['Z'] = 3;

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
      A_row = std::strtoull((argv[loop_count + 1]), &end, 10);

    else if (!(cmd_argument.compare("-A_column")))
      A_col = std::strtoull((argv[loop_count + 1]), &end, 10);

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

  B_row = A_row;
  B_col = A_col;
  C_row = A_row;
  C_col = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real,
		                                alpha_imaginary, beta_real, beta_imaginary);
  
  return 0;
}
