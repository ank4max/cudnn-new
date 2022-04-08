#include <unordered_map>
#include "cublas_gemmBatched_test.h"

template<class T>
GemmBatched<T>::GemmBatched(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), batch_count(batch_count), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void GemmBatched<T>::FreeMemory() {
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
int GemmBatched<T>::GemmBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostMatrixB = new T*[batch_count];
   HostMatrixC = new T*[batch_count];

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
      util::InitializeBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);

      break;
    }

    case 'H': {
      util::InitializeBatchedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<__half>((__half **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeBatchedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<__half>((__half **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);

      break;
    }

  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];
  HostPtrToDeviceMatB = new T*[batch_count];
  HostPtrToDeviceMatC = new T*[batch_count];

  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatB[batch], B_row * B_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (B) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatC[batch], C_row * C_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (C) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
  }

  cudaStatus = cudaMalloc((void**)&DeviceMatrixA, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void**)&DeviceMatrixB, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for matrix (B) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void**)&DeviceMatrixC, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for matrix (C) failed\n";
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
  
  //! Setting the values of matrices on device
  cudaStatus = cudaMemcpy(DeviceMatrixA, HostPtrToDeviceMatA, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Memory copy on device for matrix (A) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceMatrixB, HostPtrToDeviceMatB, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Memory copy on device for matrix (B) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceMatrixC, HostPtrToDeviceMatC, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Memory copy on device for matrix (C) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasSetMatrix(A_row, A_col, sizeof(T), HostMatrixA[batch], A_row, HostPtrToDeviceMatA[batch], A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix A failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    status = cublasSetMatrix(B_row, B_col, sizeof(T), HostMatrixB[batch], B_row, HostPtrToDeviceMatB[batch], B_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix B failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
    
    status = cublasSetMatrix(C_row, C_col, sizeof(T), HostMatrixC[batch], C_row, HostPtrToDeviceMatC[batch], C_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix C failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
  }

  
  
  /**
   * API call to performs matrix - matrix multiplication in batches : \f$ C = alpha * A[i] * B[i] + beta * C[i] \f$ \n
   * Note: C[i] matrices must not overlap, i.e. the individual gemm operations must be computable independently \n
            otherwise, undefined behavior is expected. \n
   */
    
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n, k, batchCount < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sgemmbatched API\n";
      clk_start = clock();
 
      status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  A_row, B_col, A_col, (float *)&alpha, 
                                  (float**)DeviceMatrixA, A_row, (float**)DeviceMatrixB, 
                                   B_row, (float *)&beta, (float **)DeviceMatrixC, 
                                   C_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Sgemmbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sgemmbatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dgemmbatched API\n";
      clk_start = clock();

      status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  A_row, B_col, A_col, (double *)&alpha, 
                                  (double**)DeviceMatrixA, A_row, (double**)DeviceMatrixB,
                                  B_row, (double *)&beta, (double **)DeviceMatrixC, 
                                  C_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dgemmbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dgemmbatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cgemmbatched API\n";
      clk_start = clock();
       
      status = cublasCgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  A_row, B_col, A_col, (cuComplex *)&alpha, 
                                  (cuComplex **)DeviceMatrixA, A_row, 
                                  (cuComplex **)DeviceMatrixB, B_row,
                                  (cuComplex *)&beta, (cuComplex **)DeviceMatrixC, 
                                   C_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Cgemmbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgemmbatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zgemmbatched API\n";
      clk_start = clock();
   
      status = cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  A_row, B_col, A_col, (cuDoubleComplex *)&alpha, 
                                  (cuDoubleComplex **)DeviceMatrixA, A_row, 
                                  (cuDoubleComplex **)DeviceMatrixB, B_row,
                                  (cuDoubleComplex *)&beta, (cuDoubleComplex **)DeviceMatrixC,
                                  C_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zgemmbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zgemmbatched API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Hgemmbatched API\n";
      clk_start = clock();
 
      status = cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  A_row, B_col, A_col, (__half *)&alpha, 
                                  (__half **)DeviceMatrixA, A_row, (__half **)DeviceMatrixB, 
                                   B_row, (__half *)&beta, (__half **)DeviceMatrixC, 
                                   C_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Hgemmbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Hgemmbatched API call ended\n";
      break;
    }

  }
  
  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetMatrix(C_row, C_col, sizeof(T), HostPtrToDeviceMatC[batch], 
                             C_row, HostMatrixC[batch], C_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }
  }


  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output matrix C from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix C after " << mode << "gemmbatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'D': {
      util::PrintBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'C': {
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'H': {
      util::PrintBatchedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);
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

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  GemmBatched<float> Sgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'S');
  return Sgemmbatched.GemmBatchedApiCall();
}

int mode_D(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, double alpha_real, double alpha_imaginary, 
            double beta_real, double beta_imaginary) {
            
  double alpha = alpha_real;
  double beta = beta_real;

  GemmBatched<double> Dgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'D');
  return Dgemmbatched.GemmBatchedApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  GemmBatched<cuComplex> Cgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'C');
  return Cgemmbatched.GemmBatchedApiCall();

}

int mode_Z(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  GemmBatched<cuDoubleComplex> Zgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'Z');
  return Zgemmbatched.GemmBatchedApiCall();

  
}

int mode_H(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  __half alpha = (__half) alpha_real;
  __half beta = (__half) beta_real;

  GemmBatched<__half> Hgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'H');
  return Hgemmbatched.GemmBatchedApiCall();
}

int (*cublas_func_ptr[])(int, int, int, int, int, int, int, double, double,   double, double) = {
  mode_S, mode_D, mode_C, mode_Z, mode_H
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['Z'] = 3;
  mode_index['H'] = 4;

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
    
    else if (!(cmd_argument.compare("-batch_count"))) 
      batch_count = atoi(argv[loop_count + 1]);

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
  if (A_row <= 0 || A_col <= 0 || B_col <= 0 || batch_count <= 0){
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  //! initializing values for matrix B and C
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col,   C_row, C_col, batch_count, 
                                                alpha_real, alpha_imaginary, beta_real, beta_imaginary);
  
  return status;
}
