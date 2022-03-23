#include <unordered_map>
#include "trsm.h"
#include <bits/stdc++.h>

template<class T>
Trsm<T>::Trsm(size_t A_row, size_t A_col, size_t B_row, size_t B_col, T alpha, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      alpha(alpha), mode(mode) {}

template<class T>
void Trsm<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostMatrixB)
    delete[] HostMatrixB;

  //! Destroy CuBLAS context
  status  = cublasXtDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Trsm<T>::TrsmApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostMatrixB = new T[B_row * B_col];

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

  /**
   * Switch Case - To Initialize and Print input matrices based on mode passed,
   *  A is a Triangular Matrix,
   *  B is a Normal Matrix
   */
  switch (mode) {
    case 'S': {
      util::InitializeTriangularMatrixXt<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrixXt<float>((float *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularMatrixXt<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrixXt<float>((float *)HostMatrixB, B_row, B_col);
      break;
    }

    case 'D': {
      util::InitializeTriangularMatrixXt<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrixXt<double>((double *)HostMatrixB, B_row, B_col);
      
      std::cout << "\nMatrix A:\n";
      util::PrintTriangularMatrixXt<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrixXt<double>((double *)HostMatrixB, B_row, B_col);
      break;
    }
            
    case 'C': {
      util::InitializeTriangularComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      break;
    }
        
    case 'Z': {
      util::InitializeTriangularComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
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
    return EXIT_FAILURE;   
  }

  /**
   * API call to Solves Triangular linear system with multiple right-hand-sides : \f$ A * X = alpha * B \f$
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling XtStrsm API\n";
      clk_start = clock();

      status = cublasXtStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (float *)&alpha, (float *)HostMatrixA, A_row,
                           (float *)HostMatrixB, B_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtStrsm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtStrsm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling XtDtrsm API\n";
      clk_start = clock();

      status = cublasXtDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (double *)&alpha, (double *)HostMatrixA, A_row,
                           (double *)HostMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtDtrsm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtDtrsm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling XtCtrsm API\n";
      clk_start = clock();

      status = cublasXtCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuComplex *)&alpha, (cuComplex *)HostMatrixA, A_row,
                           (cuComplex *)HostMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtCtrsm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtCtrsm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling XtZtrsm API\n";
      clk_start = clock();

      status = cublasXtZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)HostMatrixA, A_row,
                           (cuDoubleComplex *)HostMatrixB, B_row);
     
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  XtZtrsm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "XtZtrsm API call ended\n";
      break;
    }

  }

  std::cout << "\nMatrix X after Xt" << mode << "trsm operation is:\n";

  //! Print the final resultant Matrix B
  switch (mode) {
    case 'S': {
      util::PrintMatrixXt<float>((float *)HostMatrixB, B_row, B_col); 
      break;
    }

    case 'D': {
      util::PrintMatrixXt<double>((double *)HostMatrixB, B_row, B_col);  
      break;
    }

    case 'C': {
      util::PrintComplexMatrixXt<cuComplex>((cuComplex *)HostMatrixB, B_row ,B_col); 
      break;
    }

    case 'Z': {
      util::PrintComplexMatrixXt<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row ,B_col); 
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

void mode_S(size_t A_row, size_t A_col, size_t B_row, size_t B_col, double alpha_real, double alpha_imaginary) {
  
  float alpha = (float)alpha_real;
  Trsm<float> Strsm(A_row, A_col, B_row, B_col, alpha, 'S');
  Strsm.TrsmApiCall();
}

void mode_D(size_t A_row, size_t A_col, size_t B_row, size_t B_col, double alpha_real, double alpha_imaginary) {
  
  double alpha = alpha_real;
  Trsm<double> Dtrsm(A_row, A_col, B_row, B_col, alpha, 'D');
  Dtrsm.TrsmApiCall();
}

void mode_C(size_t A_row, size_t A_col, size_t B_row, size_t B_col, double alpha_real, double alpha_imaginary) {
  
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  Trsm<cuComplex> Ctrsm(A_row, A_col, B_row, B_col, alpha, 'C');
  Ctrsm.TrsmApiCall();

}

void mode_Z(size_t A_row, size_t A_col, size_t B_row, size_t B_col, double alpha_real, double alpha_imaginary) {
  
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  Trsm<cuDoubleComplex> Ztrsm(A_row, A_col, B_row, B_col, alpha, 'Z');
  Ztrsm.TrsmApiCall();
}

void (*cublas_func_ptr[])(int, int, int, int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  size_t A_row, A_col, B_row, B_col;
  double alpha_real, alpha_imaginary;
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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = std::strtoull((argv[loop_count + 1]), &end, 10);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  A_col = A_row;
  B_row = A_col;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, alpha_real, alpha_imaginary);
  
  return 0;
}
