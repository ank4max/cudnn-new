#include "cublas_symm_test.h"
#include <unordered_map>

template<class T>
Symm<T>::Symm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Symm<T>::FreeMemory() {
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

  //! Destroy events
  cudaStatus = cudaEventDestroy(start);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Unable to destroy start event\n";
  }

  cudaStatus = cudaEventDestroy(stop);
  if (cudaStatus != cudaSuccess) {
    std::cout << "Unable to destroy stop event\n";
  }
}

template<class T>
int Symm<T>::SymmApiCall() {
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
   * A is a symmetric Matrix,
   * B and C are general Matrices
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<float>((float *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      break;

    }

    case 'D': {
      util::InitializeSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<double>((double *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<double>((double *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);
      break;

    }

    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      break;

    }

    case 'Z': {
      util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
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

  cudaStatus = cudaEventCreate(&start);
   if(cudaStatus != cudaSuccess) {
     std::cout << " Failed to Create start event " << std::endl;
     FreeMemory();
    return EXIT_FAILURE;
   }
   
   cudaStatus = cudaEventCreate(&stop);
   if(cudaStatus != cudaSuccess) {
     std::cout << " Failed to create stop event " << std::endl;
     FreeMemory();
     return EXIT_FAILURE;
   }

   float milliseconds;

  /**
   * API call to performs symmetric matrix - matrix multiplication : \f$ C = alpha * A * B + beta * C \f$
   */
   
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully  \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized  \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n < 0  \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */ 
   
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Ssymm API\n";

      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           A_row, B_col, (float *)&alpha, (float *)DeviceMatrixA,
                           A_row, (float *)DeviceMatrixB, B_row, (float *)&beta, 
                           (float *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ssymm kernel execution error\n";
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
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ssymm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for C " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
        
      

      
      std::cout << "Ssymm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dsymm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           A_row, B_col, (double *)&alpha, (double *)DeviceMatrixA,
                           A_row, (double *)DeviceMatrixB, B_row, (double *)&beta, 
                           (double *)DeviceMatrixC, C_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dsymm kernel execution error\n";
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
      std::cout << "Dsymm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Csymm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           A_row, B_col, (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA,
                           A_row, (cuComplex *)DeviceMatrixB, B_row, (cuComplex *)&beta, 
                           (cuComplex *)DeviceMatrixC, C_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Csymm kernel execution error\n";
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
      std::cout << "Csymm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zsymm API\n";
      cudaStatus = cudaEventRecord(start);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           A_row, B_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA,
                           A_row, (cuDoubleComplex *)DeviceMatrixB, B_row, (cuDoubleComplex *)&beta, 
                           (cuDoubleComplex *)DeviceMatrixC, C_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zsymm kernel execution error\n";
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
      std::cout << "Zsymm API call ended\n";
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

  std::cout << "\nMatrix C after " << mode << "symm operation is:\n";

  //! Print the final resultant Matrix C
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
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
      break;
    }
  }

  //! Execution time for API
  cudaStatus = cudaEventElapsedTime(&milliseconds, start, stop);
  if(cudaStatus != cudaSuccess) {
    std::cout << " Failed to store API execution time " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  long long total_operations = 1ULL * A_row * A_col * B_col;
  double seconds = SECONDS(milliseconds);

  //! Print Latency and Throughput of the API
  std::cout << "\nLatency: " <<  milliseconds <<
               "\nThroughput: " << THROUGHPUT(seconds, total_operations) << "\n\n";

  FreeMemory();
  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Symm<float> Ssymm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'S');
  return Ssymm.SymmApiCall();

}

int mode_D(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  double alpha = alpha_real;
  double beta = beta_real;

  Symm<double> Dsymm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'D');
  return Dsymm.SymmApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Symm<cuComplex> Csymm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'C');
  return Csymm.SymmApiCall();
}

int mode_Z(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Symm<cuDoubleComplex> Zsymm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'Z');
  return Zsymm.SymmApiCall();
}


int (*cublas_func_ptr[])(int, int, int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
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
    
    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Dimension check
  if(A_row <= 0 || B_col <= 0) {
      std::cout << "Minimum Dimension error\n";
      return EXIT_FAILURE;
  }



  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, 
		                                alpha_real, alpha_imaginary, beta_real, beta_imaginary); 

  return status;
}
