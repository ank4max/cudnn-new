#include <unordered_map>
#include "cublas_syrk_test.h"

template<class T>
Syrk<T>::Syrk(int A_row, int A_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Syrk<T>::FreeMemory() {
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
int Syrk<T>::SyrkApiCall() {
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
   * A is a general Matrix,
   * C is a symmetric Matrix
   */
  switch (mode) {
    case 'S': {
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'D': {
      util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix C:\n";
      util::PrintSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);
      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col); 
      break;  
    }

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

  //! Event create for storing time
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

   float milliseconds = 0;

  /**
   * API call to performs the symmetric rank- k update : \f$ C = alpha * A * A^T + beta * C \f$
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
      std::cout << "\nCalling Ssyrk API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, (float *)&alpha,
                           (float *)DeviceMatrixA, A_row, (float *)&beta,
                           (float *)DeviceMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ssyrk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop, 0);
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
       
      
      std::cout << "Ssyrk API call ended\n";
      break;
    }
                            
    case 'D': {
      std::cout << "\nCalling Dsyrk API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, (double *)&alpha,
                           (double *)DeviceMatrixA, A_row, (double *)&beta,
                           (double *)DeviceMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dsyrk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop, 0);
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

      
      std::cout << "Dsyrk API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Csyrk API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      status = cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, (cuComplex *)&alpha,
                           (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)&beta,
                           (cuComplex *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Csyrk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop, 0);
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
      std::cout << "Csyrk API call ended\n";
      break;
    }
      
    case 'Z': {
      std::cout << "\nCalling Zsyrk API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           A_row, A_col, (cuDoubleComplex *)&alpha,
                           (cuDoubleComplex *)DeviceMatrixA, A_row,
                           (cuDoubleComplex *)&beta,
                           (cuDoubleComplex *)DeviceMatrixC, C_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zsyrk kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaEventRecord(stop, 0);
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
      std::cout << "Zsyrk API call ended\n";
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

  std::cout << "\nMatrix C after " << mode << "syrk operation is:\n";

  //! Print the final resultant Matrix C
  switch (mode) {
    case 'S': {
      util::PrintSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col); 
      break;
    }

    case 'D': {
      util::PrintSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);  
      break;
    }

    case 'C': {
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }

    case 'Z': {
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
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

  long long total_operations = 1ULL * A_row * A_col * C_col;
  double seconds = SECONDS(milliseconds);

  //! Print Latency and Throughput of the API
   
  std::cout << "\nLatency: " <<  milliseconds <<
               "\nThroughput: " << THROUGHPUT(seconds, total_operations) << "\n\n";

  FreeMemory();
  return EXIT_SUCCESS;

}

int mode_S(int A_row, int A_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Syrk<float> Ssyrk(A_row, A_col, C_row, C_col, alpha, beta, 'S');
  return Ssyrk.SyrkApiCall();
}

int mode_D(int A_row, int A_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  double alpha = alpha_real;
  double beta = beta_real;

  Syrk<double> Dsyrk(A_row, A_col, C_row, C_col, alpha, beta, 'D');
  return Dsyrk.SyrkApiCall();
}

int mode_C(int A_row, int A_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Syrk<cuComplex> Csyrk(A_row, A_col, C_row, C_col, alpha, beta, 'C');
  return Csyrk.SyrkApiCall();
}

int mode_Z(int A_row, int A_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
    
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Syrk<cuDoubleComplex> Zsyrk(A_row, A_col, C_row, C_col, alpha, beta, 'Z');
  return Zsyrk.SyrkApiCall();

}

int (*cublas_func_ptr[])(int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, C_row, C_col, status;
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

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

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
  if(A_row <= 0 || A_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }
  
  C_row = A_row;
  C_col = A_row;
  
  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, alpha_real, 
		                                alpha_imaginary, beta_real, beta_imaginary);

  return status;
}
