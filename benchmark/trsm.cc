#include <unordered_map>
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
 
  //! Event create for storing time
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

   float milliseconds = 0;


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
      std::cout << "\nCalling Strsm API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (float *)&alpha, (float *)DeviceMatrixA, A_row,
                           (float *)DeviceMatrixB, B_row);
        
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Strsm kernel execution error\n";
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
      std::cout << "Strsm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtrsm API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (double *)&alpha, (double *)DeviceMatrixA, A_row,
                           (double *)DeviceMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dtrsm kernel execution error\n";
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
      std::cout << "Dtrsm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctrsm API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceMatrixB, B_row);
      
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ctrsm kernel execution error\n";
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
      std::cout << "Ctrsm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztrsm API\n";
      cudaStatus = cudaEventRecord(start, 0);
      if(cudaStatus != cudaSuccess) {
        std::cout << " Failed to record start time " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA, A_row,
                           (cuDoubleComplex *)DeviceMatrixB, B_row);
     
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ztrsm kernel execution error\n";
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
      std::cout << "Ztrsm API call ended\n";
      break;
    }

  }

  //! Copy Matrix B, holding resultant matrix, from Device to Host using cublasGetMatrix()

  status = cublasGetMatrix(B_row, B_col, sizeof(*HostMatrixB),
                           DeviceMatrixB, B_row, HostMatrixB, B_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output matrix B from device\n";
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

int mode_S(int A_row, int A_col, int B_row, int B_col, double alpha_real, double alpha_imaginary) {
  
  float alpha = (float)alpha_real;
  Trsm<float> Strsm(A_row, A_col, B_row, B_col, alpha, 'S');
  return Strsm.TrsmApiCall();
}

int mode_D(int A_row, int A_col, int B_row, int B_col, double alpha_real, double alpha_imaginary) {
  
  double alpha = alpha_real;
  Trsm<double> Dtrsm(A_row, A_col, B_row, B_col, alpha, 'D');
  return Dtrsm.TrsmApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, double alpha_real, double alpha_imaginary) {
  
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  Trsm<cuComplex> Ctrsm(A_row, A_col, B_row, B_col, alpha, 'C');
  return Ctrsm.TrsmApiCall();

}

int mode_Z(int A_row, int A_col, int B_row, int B_col, double alpha_real, double alpha_imaginary) {
  
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  Trsm<cuDoubleComplex> Ztrsm(A_row, A_col, B_row, B_col, alpha, 'Z');
  return Ztrsm.TrsmApiCall();
}

int (*cublas_func_ptr[])(int, int, int, int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, status;
  double alpha_real, alpha_imaginary;
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

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  if(A_row <= 0 || B_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }  

  A_col = A_row;
  B_row = A_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, alpha_real, alpha_imaginary);
  
  return status;
}
