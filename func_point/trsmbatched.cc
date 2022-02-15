
#include <unordered_map>
#include "cublas_trsmbatched_test.h"

template<class T>
TrsmBatched<T>::TrsmBatched(int A_row, int A_col, int B_row, int B_col, int batch_count, T alpha, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      batch_count(batch_count), alpha(alpha), mode(mode) {}

template<class T>
void TrsmBatched<T>::FreeMemory() {
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
int TrsmBatched<T>::TrsmBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostMatrixB = new T*[batch_count];

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
   * A is a triangular Matrix  
   * B is a general matrix
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeBatchedTriangularMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedTriangularMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      break;
    }

    case 'C': {
      util::InitializeBatchedTriangularComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      break;
    }

    case 'Z': {
       util::InitializeBatchedTriangularComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
       util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);

       std::cout << "\nMatrix A:\n";
       util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
       std::cout << "\nMatrix B:\n";
       util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
       break;
    }

  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];
  HostPtrToDeviceMatB = new T*[batch_count];
  

  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      fprintf (stderr, "!!!! Device memory allocation for matrix (A) failed\n");
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatB[batch], B_row * B_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      fprintf (stderr, "!!!! Device memory allocation for matrix (B) failed\n");
      FreeMemory();
      return EXIT_FAILURE;
    }
  }

  cudaStatus = cudaMalloc((void**)&DeviceMatrixA, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Device memory allocation for matrix (A) failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void**)&DeviceMatrixB, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Device memory allocation for matrix (B) failed\n");
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
  
  //! Setting the values of matrices on device
  cudaStatus = cudaMemcpy(DeviceMatrixA, HostPtrToDeviceMatA, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Memory copy on device for matrix (A) failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceMatrixB, HostPtrToDeviceMatB, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Memory copy on device for matrix (B) failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  
  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasSetMatrix(A_row, A_col, sizeof(T), HostMatrixA[batch], A_row, HostPtrToDeviceMatA[batch], A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! Setting up values on device for Matrix A failed\n");
      FreeMemory();
      return EXIT_FAILURE;
    }

    status = cublasSetMatrix(B_row, B_col, sizeof(T), HostMatrixB[batch], B_row, HostPtrToDeviceMatB[batch], B_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! Setting up values on device for Matrix B failed\n");
      FreeMemory();
      return EXIT_FAILURE;
    }
  }
  
  /**
   * API call to performs matrix - matrix multiplication in batches : C = alpha * A[i] * B[i] + beta * C[i]
   * This function works for any sizes but is intended to be used for matrices of small sizes where the launch overhead is a significant factor. 
   * For bigger sizes, it might be advantageous to call batchCount times the regular cublas<t>trsm within a set of CUDA streams.
   * The current implementation is limited to devices with compute capability above or equal 2.0.
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below :
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n <0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Strsmbatched API\n";
      clk_start = clock();
          
      status = cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                                  (const float *)&alpha, (float* const*)DeviceMatrixA, A_row, 
                                  (float* const*)DeviceMatrixB, B_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Strsmbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Strsmbatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtrsmbatched API\n";
      clk_start = clock();

      status = cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                  CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                  (const double *)&alpha, (double* const*)DeviceMatrixA, A_row, 
                                  (double* const*)DeviceMatrixB, B_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dtrsmbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtrsmbatched API call ended\n";
      break;
    }
          
    case 'C': {
      std::cout << "\nCalling Ctrsmbatched API\n";
      clk_start = clock();
       
      status = cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                  CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                  (const cuComplex *)&alpha, 
                                  (cuComplex* const*)DeviceMatrixA, A_row, 
                                  (cuComplex* const*)DeviceMatrixB, B_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ctrsmbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctrsmbatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztrsmbatched API\n";
      clk_start = clock();
   
      status = cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                  CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                  (const cuDoubleComplex *)&alpha, 
                                  (cuDoubleComplex* const*)DeviceMatrixA, A_row, 
                                  (cuDoubleComplex* const*)DeviceMatrixB, B_row, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ztrsmbatched kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztrsmbatched API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetMatrix(B_row, B_col, sizeof(T), HostPtrToDeviceMatB[batch], 
                             B_row, HostMatrixB[batch], B_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! API execution failed\n");
      return EXIT_FAILURE;
    }
  }
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix B after " << mode << "trsmbatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      break;
    }

    case 'D': {
      util::PrintBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      break;
    }

    case 'C': {
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      break;
     }

    case 'Z': {
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
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

void mode_S(int A_row, int A_col, int B_row, int B_col, int batch_count, double alpha_real, double alpha_imaginary) {
       
  float alpha = (float)alpha_real;
  TrsmBatched<float> Strsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, 'S');
  Strsmbatched.TrsmBatchedApiCall();
}

void mode_D(int A_row, int A_col, int B_row, int B_col, int batch_count, double alpha_real, double alpha_imaginary) {
            
  double alpha = alpha_real;
  TrsmBatched<double> Dtrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, 'D');
  Dtrsmbatched.TrsmBatchedApiCall();

}

void mode_C(int A_row, int A_col, int B_row, int B_col, int batch_count, double alpha_real, double alpha_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  TrsmBatched<cuComplex> Ctrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, 'C');
  Ctrsmbatched.TrsmBatchedApiCall();
  
}

void mode_Z(int A_row, int A_col, int B_row, int B_col, int batch_count, double alpha_real, double alpha_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  TrsmBatched<cuDoubleComplex> Ztrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, 'Z');
  Ztrsmbatched.TrsmBatchedApiCall();
  
}

void (*cublas_func_ptr[])(int, int, int, int, int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, batch_count, status;
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

  //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-batch_count"))) 
      batch_count = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! Initializing values for matrix A and B
 A_col = A_row;
 B_row = A_col;

(*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, batch_count, alpha_real, alpha_imaginary);
  

  return EXIT_SUCCESS;
}


  
