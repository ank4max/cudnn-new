%%writefile max.cc
#include "cublas_gemmStridedBatched_test.h"

template<class T>
GemmStridedBatched<T>::GemmStridedBatched(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), batch_count(batch_count), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void GemmStridedBatched<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int GemmStridedBatched<T>::GemmStridedBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T[batch_count * A_row * A_col];
   HostMatrixB = new T[batch_count * B_row * B_col];
   HostMatrixC = new T[batch_count * C_row * C_col];

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

  if (!HostMatrixC) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  // initialize CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }


  /**
   * Switch Case - To Initialize and Print input matrices based on mode passed,
   * A, B and C are general matrices
   */
  
  switch (mode) {
     case 'S': {
       util::InitializeStridedBatchedMatrix<float>((float *)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeStridedBatchedMatrix<float>((float *)HostMatrixB, B_row, B_col, batch_count);
       util::InitializeStridedBatchedMatrix<float>((float *)HostMatrixC, C_row, C_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintStridedBatchedMatrix<float>((float *)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintStridedBatchedMatrix<float>((float *)HostMatrixB, B_row, B_col, batch_count);
       std::cout << "\nMatrix C:\n";
       util::PrintStridedBatchedMatrix<float>((float *)HostMatrixC, C_row, C_col, batch_count);          

       //allocating matrices on device    
       status = cublasAlloc(batch_count * A_row * A_col, sizeof(float), (void**)&DeviceMatrixA);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (A)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(float), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (B)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(float), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (C)\n");
         return EXIT_FAILURE;
       }

       break;
     }

     case 'D': {
       util::InitializeStridedBatchedMatrix<double>((double *)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeStridedBatchedMatrix<double>((double *)HostMatrixB, B_row, B_col, batch_count);
       util::InitializeStridedBatchedMatrix<double>((double *)HostMatrixC, C_row, C_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintStridedBatchedMatrix<double>((double *)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintStridedBatchedMatrix<double>((double *)HostMatrixB, B_row, B_col, batch_count);
       std::cout << "\nMatrix C:\n";
       util::PrintStridedBatchedMatrix<double>((double *)HostMatrixC, C_row, C_col, batch_count);

       //allocating matrices on device    
       status = cublasAlloc(batch_count * A_row * A_col, sizeof(double), (void**)&DeviceMatrixA);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (A)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(double), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (B)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(double), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (C)\n");
         return EXIT_FAILURE;
       }

       break;
     }

     case 'C': {
       util::InitializeStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col, batch_count);
       util::InitializeStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col, batch_count);
       std::cout << "\nMatrix C:\n";
       util::PrintStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col, batch_count);

       //allocating matrices on device    
       status = cublasAlloc(batch_count * A_row * A_col, sizeof(cuComplex), (void**)&DeviceMatrixA);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (A)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(cuComplex), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (B)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(cuComplex), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (C)\n");
         return EXIT_FAILURE;
       }

       break;
     }

     case 'Z': {
       util::InitializeStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col, batch_count);
       util::InitializeStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col, batch_count);
       std::cout << "\nMatrix C:\n";
       util::PrintStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col, batch_count);

       //allocating matrices on device    
       status = cublasAlloc(batch_count * A_row * A_col, sizeof(cuDoubleComplex), (void**)&DeviceMatrixA);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (A)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(cuDoubleComplex), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (B)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(cuDoubleComplex), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (C)\n");
         return EXIT_FAILURE;
       }

       break;
     }

     case 'H': {
       util::InitializeStridedBatchedMatrix<__half>((__half *)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeStridedBatchedMatrix<__half>((__half *)HostMatrixB, B_row, B_col, batch_count);
       util::InitializeStridedBatchedMatrix<__half>((__half *)HostMatrixC, C_row, C_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintStridedBatchedMatrix<__half>((__half *)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintStridedBatchedMatrix<__half>((__half *)HostMatrixB, B_row, B_col, batch_count);
       std::cout << "\nMatrix C:\n";
       util::PrintStridedBatchedMatrix<__half>((__half *)HostMatrixC, C_row, C_col, batch_count);
   
       status = cublasAlloc(batch_count * A_row * A_col, sizeof(__half), (void**)&DeviceMatrixA);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (A)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(__half), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (B)\n");
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(__half), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! Device memory allocation error (C)\n");
         return EXIT_FAILURE;
       }

       break;
     }
   }
   
   // setting the values of matrices on device
   cudaStatus = cudaMemcpy(DeviceMatrixA, HostMatrixA, sizeof(T) * batch_count * A_row * A_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
     return EXIT_FAILURE;
   }
   cudaStatus = cudaMemcpy(DeviceMatrixB, HostMatrixB, sizeof(T) * batch_count * B_row * B_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     fprintf (stderr, "!!!! Setting up values on device for matrix (B) failed\n");
     return EXIT_FAILURE;
   }
   cudaStatus = cudaMemcpy(DeviceMatrixC, HostMatrixC, sizeof(T) * batch_count * C_row * C_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     fprintf (stderr, "!!!! Setting up values on device for matrix (C) failed\n");
     return EXIT_FAILURE;
   }
   
   // defining stride to differentiate between each batch
   long long int strideA = A_row * A_col;
   long long int strideB = B_row * B_col;
   long long int strideC = C_row * C_col;
  
  /**
   * API call to performs matrix - matrix multiplication : C[i] = alpha * A[i] * B[i] + beta * C[i]
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sgemmstridedbatched API\n";
      clk_start = clock();
 
      status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (float *)&alpha, 
                                         (float *)DeviceMatrixA, A_row, strideA,
                                         (float *)DeviceMatrixB, B_row, strideB,
                                         (float *)&beta, (float *)DeviceMatrixC, 
                                         C_row, strideC, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sgemmstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sgemmstridedbatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dgemmstridedbatched API\n";
      clk_start = clock();

      status = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (double *)&alpha, 
                                         (double *)DeviceMatrixA, A_row, strideA,
                                         (double *)DeviceMatrixB, B_row, strideB,
                                         (double *)&beta, (double *)DeviceMatrixC, 
                                         C_row, strideC, batch_count);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dgemmstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dgemmstridedbatched API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Hgemmstridedbatched API\n";
      clk_start = clock();

      status = cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (__half *)&alpha,
                                         (__half *)DeviceMatrixA, A_row, strideA,
                                         (__half *)DeviceMatrixB, B_row, strideB,
                                         (__half *)&beta, (__half *)DeviceMatrixC,
                                         C_row, strideC, batch_count);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Hgemmstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Hgemmstridedbatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cgemmstridedbatched API\n";
      clk_start = clock();
       
      status = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (cuComplex *)&alpha, 
                                         (cuComplex *)DeviceMatrixA, A_row, strideA, 
                                         (cuComplex *)DeviceMatrixB, B_row, strideB,
                                         (cuComplex *)&beta, (cuComplex *)DeviceMatrixC, 
                                         C_row, strideC, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cgemmstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgemmstridedbatched API call ended\n";
      break;
    }

    case '3': {
      std::cout << "\nCalling Cgemm3mstridedbatched API\n";
      clk_start = clock();
       
      status = cublasCgemm3mStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           A_row, B_col, A_col, (cuComplex *)&alpha,
                                           (cuComplex *)DeviceMatrixA, A_row, strideA,
                                           (cuComplex *)DeviceMatrixB, B_row, strideB,
                                           (cuComplex *)&beta, (cuComplex *)DeviceMatrixC,
                                           C_row, strideC, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cgemm3mstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgemm3mstridedbatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zgemmstridedbatched API\n";
      clk_start = clock();
   
      status = cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (cuDoubleComplex *)&alpha,
                                         (cuDoubleComplex *)DeviceMatrixA, A_row, strideA, 
                                         (cuDoubleComplex *)DeviceMatrixB, B_row, strideB,
                                         (cuDoubleComplex *)&beta, (cuDoubleComplex *)DeviceMatrixC, 
                                         C_row, strideC, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zgemmstridedbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zgemmstridedbatched API call ended\n";
      break;
    }

  }
  
  //! Getting the final output
  cudaStatus = cudaMemcpy(HostMatrixC, DeviceMatrixC,  sizeof(T) * batch_count * C_row * C_col, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Failed to to Get values in Host Matrix C");
    return EXIT_FAILURE;
  }


  std::cout << "\nMatrix C after " << mode << "gemmstridedbatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintStridedBatchedMatrix<float>((float *)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'D': {
      util::PrintStridedBatchedMatrix<double>((double *)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'C': {
      util::PrintStridedBatchedComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'Z': {
      util::PrintStridedBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'H': {
      util::PrintStridedBatchedMatrix<__half>((__half *)HostMatrixC, C_row, C_col, batch_count);
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

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;

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
  
  //! initializing values for matrix B and C
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  //! Calling GemmStridedBatched API based on mode
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;
      float beta = (float)beta_real;
      GemmStridedBatched<float> Sgemmstridedbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Sgemmstridedbatched.GemmStridedBatchedApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;
      double beta = beta_real;
      GemmStridedBatched<double> Dgemmstridedbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Dgemmstridedbatched.GemmStridedBatchedApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
      cuComplex beta = {(float)beta_real, (float)beta_imaginary};
      GemmStridedBatched<cuComplex> Cgemmstridedbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Cgemmstridedbatched.GemmStridedBatchedApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
      cuDoubleComplex beta = {beta_real, beta_imaginary};
      GemmStridedBatched<cuDoubleComplex> Zgemmstridedbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Zgemmstridedbatched.GemmStridedBatchedApiCall();
      break;
    }

    case 'H': {
      __half alpha = (__half)alpha_real;
      __half beta = (__half)beta_real;
      GemmStridedBatched<__half> Hgemmstridedbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Hgemmstridedbatched.GemmStridedBatchedApiCall();
      break;
    }

  }

  return EXIT_SUCCESS;
}


  
