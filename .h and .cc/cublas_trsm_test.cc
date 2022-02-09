#include "cublas_trsm_test.h"

template<class T>
Trsm<T>::Trsm(int A_row, int A_col, int B_row, int B_col, T alpha, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      alpha(alpha), mode(mode) {}

template<class T>
void Trsm<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostMatrixB)
    delete[] HostMatrixB;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixB);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Trsm<T>::TrsmApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostMatrixB = new T[B_row * B_col];

  if (!HostMatrixA) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostMatrixB) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixB)\n");
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
      util::InitializeTriangularMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
      break;
    }

    case 'D': {
      util::InitializeTriangularMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<double>((double *)HostMatrixB, B_row, B_col);
      
      std::cout << "\nMatrix A:\n";
      util::PrintTriangularMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col);
      break;
    }
            
    case 'C': {
      util::InitializeTriangularComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      break;
    }
        
    case 'Z': {
      util::InitializeTriangularComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);

      std::cout << "\nMatrix A:\n";
      util::PrintTriangularComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
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

  status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * API call to Solves Triangular linear system with multiple right-hand-sides :  A * X = alpha * B
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Strsm API\n";
      clk_start = clock();

      status = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (float *)&alpha, (float *)DeviceMatrixA, A_row,
                           (float *)DeviceMatrixB, B_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Strsm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Strsm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtrsm API\n";
      clk_start = clock();

      status = cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (double *)&alpha, (double *)DeviceMatrixA, A_row,
                           (double *)DeviceMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dtrsm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtrsm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctrsm API\n";
      clk_start = clock();

      status = cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ctrsm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctrsm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztrsm API\n";
      clk_start = clock();

      status = cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA, A_row,
                           (cuDoubleComplex *)DeviceMatrixB, B_row);
     
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ztrsm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztrsm API call ended\n";
      break;
    }

  }

  //! Copy Matrix B, holding resultant matrix, from Device to Host using cublasGetMatrix()

  status = cublasGetMatrix(B_row, B_col, sizeof(*HostMatrixB),
                           DeviceMatrixB, B_row, HostMatrixB, B_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix B from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix X after " << mode << "trsm operation is:\n";

  //! Print the final resultant Matrix B
  switch (mode) {
    case 'S': {
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col); 
      break;
    }

    case 'D': {
      util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col);  
      break;
    }

    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row ,B_col); 
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row ,B_col); 
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

int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, status;
  double alpha_real, alpha_imaginary;
  char mode;

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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  A_col = A_row;
  B_row = A_col;

  //! Calling Trsm API based on mode
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;

      Trsm<float> Strsm(A_row, A_col, B_row, B_col, alpha, mode);
      status = Strsm.TrsmApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;

      Trsm<double> Dtrsm(A_row, A_col, B_row, B_col, alpha, mode);
      status = Dtrsm.TrsmApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

      Trsm<cuComplex> Ctrsm(A_row, A_col, B_row, B_col, alpha, mode);
      status = Ctrsm.TrsmApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

      Trsm<cuDoubleComplex> Ztrsm(A_row, A_col, B_row, B_col, alpha, mode);
      status = Ztrsm.TrsmApiCall();
      break;
    }

  }

  return EXIT_SUCCESS;
}
