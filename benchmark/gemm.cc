%%writefile gemm1apr.cc
#include <unordered_map>
#include "cublas_gemm_test.h"

template<class T>
Gemm<T>::Gemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Gemm<T>::FreeMemory() {
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
int Gemm<T>::GemmApiCall() {
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

    case 'D': {
      util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<double>((double *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<double>((double *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);
       
      break;
    }

    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);      
      
      break;
    }

    case 'H': {
      util::InitializeMatrix<__half>((__half *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<__half>((__half *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<__half>((__half *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix <__half> ((__half *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix <__half> ((__half *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix <__half> ((__half *)HostMatrixC, C_row, C_col);
      
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
  
  /**
   * API call to performs matrix - matrix multiplication : \f$ C = alpha * A * B + beta * C \f$
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n, k < 0 \n
   * CUBLAS_STATUS_ARCH_MISMATCH - In the case of cublasHgemm the device does not support math in half precision. \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */

   
   cudaStatus = cudaEventCreate(&start);
   if(cudaStatus != cudaSuccess) {
     std::cout << " Failed to Create start event " << std::endl;
     FreeMemory();
    return EXIT_FAILURE;
   }
   
   cudaStatus = cudaEventCreate(&stop);
   if(cudaStatus != cudaSuccess) {
     std::cout << " Failed to create stop time " << std::endl;
     FreeMemory();
     return EXIT_FAILURE;
   }

   float milliseconds;
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sgemm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (float *)&alpha,
                           (float *)DeviceMatrixA, A_row,
                           (float *)DeviceMatrixB, B_row, (float *)&beta,
                           (float *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Sgemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }                     

      cudaStatus = cudaEventRecord(stop);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record stop time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
      cudaStatus = cudaEventSynchronize(stop);
      if (cudaStatus != cudaSuccess) {
        std::cout << "Failed to synchronize events\n";
        FreeMemory();
        return EXIT_FAILURE;
      }
      std::cout << "Sgemm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dgemm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (double *)&alpha,
                           (double *)DeviceMatrixA, A_row,
                           (double *)DeviceMatrixB, B_row,
                           (double *)&beta,
                           (double *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dgemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record stop time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventSynchronize(stop);
      if (cudaStatus != cudaSuccess) {
        std::cout << "Failed to synchronize events\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      
      std::cout << "Dgemm API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Hgemm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (__half *)&alpha,
                           (__half *)DeviceMatrixA, A_row,
                           (__half *)DeviceMatrixB, B_row,
                           (__half *)&beta,
                           (__half *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Hgemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record stop time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventSynchronize(stop);
      if (cudaStatus != cudaSuccess) {
        std::cout << "Failed to synchronize events\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      std::cout << "Hgemm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cgemm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (cuComplex *)&alpha,
                           (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceMatrixB, B_row,
                           (cuComplex *)&beta,
                           (cuComplex *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Cgemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record stop time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventSynchronize(stop);
      if (cudaStatus != cudaSuccess) {
        std::cout << "Failed to synchronize events\n";
        FreeMemory();
        return EXIT_FAILURE;
      }
      std::cout << "Cgemm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zgemm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_row,
                           B_col, A_col, (cuDoubleComplex *)&alpha,
                           (cuDoubleComplex *)DeviceMatrixA, A_row,
                           (cuDoubleComplex *)DeviceMatrixB, B_row,
                           (cuDoubleComplex *)&beta,
                           (cuDoubleComplex *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zgemm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record stop time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventSynchronize(stop);
      if (cudaStatus != cudaSuccess) {
        std::cout << "Failed to synchronize events\n";
        FreeMemory();
        return EXIT_FAILURE;
      }
      std::cout << "Zgemm API call ended\n";
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

  std::cout << "\nMatrix C after " << mode << "gemm operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'D': {
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col);
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col);
      break;
    }

    case 'H': {
      util::PrintMatrix<__half>((__half *)HostMatrixC, C_row, C_col);
      break;
    }
  }

  long long total_operations = 1ULL * A_row * A_col * B_col;

  //! printing latency and throughput of the function
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = milliseconds/1000; 
  std::cout << "\nLatency: " <<  milliseconds <<
               "\nThroughput: " << THROUGHPUT(seconds, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Gemm<float> Sgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S' );
  return Sgemm.GemmApiCall();
}

int mode_D(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary, 
            double beta_real, double beta_imaginary) {
            
  double alpha = alpha_real;
  double beta = beta_real;

  Gemm<double> Dgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'D');
  return Dgemm.GemmApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Gemm<cuComplex> Cgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'C');
  return Cgemm.GemmApiCall();

}

int mode_Z(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Gemm<cuDoubleComplex> Zgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'Z');
  return Zgemm.GemmApiCall();

}

int mode_H(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {

  __half alpha = (__half)alpha_real;
  __half beta = (__half)beta_real;

  Gemm<__half> Hgemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'H');
  return Hgemm.GemmApiCall();
}

int (*cublas_func_ptr[])(int, int, int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z, mode_H
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, status;
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
  if(A_row <= 0 || A_col <= 0 || B_col <= 0 ) {
      std::cout << "Minimum Dimension error\n";
      return EXIT_FAILURE;
  }
  
  //! initializing values for matrix B and C
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, 
                                                alpha_imaginary, beta_real, beta_imaginary);
  
  return status;
}
