%%writefile cublas_GetrsBatched_test.cc
#include <unordered_map>
#include "cublas_GetrsBatched_test.h"

template<class T>
GetrsBatched<T>::GetrsBatched(int A_row, int A_col, int B_row, int B_col, int batch_count, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col), batch_count(batch_count), mode(mode) {}

template<class T>
void GetrsBatched<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;
  
  if (HostMatrixB)
    delete[] HostMatrixB;
  
  if (HostdevIpiv)
    delete[] HostdevIpiv;


  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixB);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Matrix B" << std::endl;
  }

  cudaStatus = cudaFree(DevicedevIpiv);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for DevicedevIpiv" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GetrsBatched<T>::GetrsBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostMatrixB = new T*[batch_count];
   HostdevIpiv =  new int[A_row * batch_count];
  

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
   * A, B and C are general matrices
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
      
      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix HostdevIpiv:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
      
      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix HostdevIpiv:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix HostdevIpiv:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
      util::InitializeMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nMatrix HostdevIpiv:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
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
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatB[batch], B_row * B_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for Matrix (B) failed\n";
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

  cudaStatus = cudaMalloc((void**)&DevicedevIpiv, A_row * batch_count * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for matrix (DevicedevIpiv) failed\n";
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
    std::cout << "!!!! Memory copy on device for TauArray failed\n";
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
  }

  status = cublasSetMatrix(A_row, batch_count, sizeof(int), HostdevIpiv, A_row, DevicedevIpiv, A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix B failed\n";
      FreeMemory();
      return EXIT_FAILURE;
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

   int *info;
   info = new int;


  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SgetrsBatched API\n";
      clk_start = clock();
 
      status = cublasSgetrsBatched(handle, CUBLAS_OP_N, A_row, B_col, (float **)DeviceMatrixA,
                                  A_row, DevicedevIpiv, (float **)DeviceMatrixB, B_row, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SgetrsBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SgetrsBatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DgetrsBatched API\n";
      clk_start = clock();

      status = cublasDgetrsBatched(handle, CUBLAS_OP_N, A_row, B_col, (double **)DeviceMatrixA,
                                  A_row, DevicedevIpiv, (double **)DeviceMatrixB, B_row, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DgetrsBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DgetrsBatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CgetrsBatched API\n";
      clk_start = clock();
       
      status = cublasCgetrsBatched(handle, CUBLAS_OP_N, A_row, B_col, (cuComplex **)DeviceMatrixA,
                                  A_row, DevicedevIpiv, (cuComplex **)DeviceMatrixB, B_row, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CgetrsBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CgetrsBatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZgetrsBatched API\n";
      clk_start = clock();
   
      status = cublasZgetrsBatched(handle, CUBLAS_OP_N, A_row, B_col, (cuDoubleComplex **)DeviceMatrixA,
                                  A_row, DevicedevIpiv, (cuDoubleComplex **)DeviceMatrixB, B_row, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZgetrsBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZgetrsBatched API call ended\n";
      break;
    }
    
  }
  
  //! Copy Matrix A and B, holding resultant matrix, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetMatrix(A_row, A_col, sizeof(T), HostPtrToDeviceMatA[batch], 
                             A_row, HostMatrixA[batch], A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }

    status = cublasGetMatrix(B_row, B_col, sizeof(T), HostPtrToDeviceMatB[batch], 
                             B_row, HostMatrixB[batch], B_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }
  }

  status = cublasGetMatrix(A_row, batch_count, sizeof(int), DevicedevIpiv, 
                             A_row, HostdevIpiv, A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }



  std::cout << "\nMatrix A after " << mode << "GetrsBatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B after " << mode << "GetrsBatched operation is:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nPivot sequencing array after " << mode << "GetrsBatched operation is:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'D': {
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B after " << mode << "GetrsBatched operation is:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nPivot sequencing array after " << mode << "GetrsBatched operation is:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);

      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'C': {
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B after " << mode << "GetrsBatched operation is:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nPivot sequencing array after" << mode << "GetrsBatched operation is:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix B after " << mode << "GetrsBatched operation is:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
      std::cout << "\nPivot sequencing array after" << mode << "GetrsBatched operation is:\n";
      util::PrintMatrix<int>((int *)HostdevIpiv, A_row, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

  }

  long long total_operations = A_row * A_col ;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col, int B_row, int B_col, int batch_count) {

  GetrsBatched<float> SGetrsBatched(A_row, A_col, B_row, B_col, batch_count,'S');
  return SGetrsBatched.GetrsBatchedApiCall();
}

int mode_D(int A_row, int A_col, int B_row, int B_col, int batch_count) {

  GetrsBatched<double> DGetrsBatched(A_row, A_col, B_row, B_col, batch_count, 'D');
  return DGetrsBatched.GetrsBatchedApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int batch_count) {

  GetrsBatched<cuComplex> CGetrsBatched(A_row, A_col, B_row, B_col, batch_count, 'C');
  return CGetrsBatched.GetrsBatchedApiCall();

}

int mode_Z(int A_row, int A_col, int B_row, int B_col, int batch_count) {

  GetrsBatched<cuDoubleComplex> ZGetrsBatched(A_row, A_col, B_row, B_col, batch_count, 'Z');
  return ZGetrsBatched.GetrsBatchedApiCall();  
}


int (*cublas_func_ptr[])(int, int, int, int, int) = {
  mode_S,mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, nhrs, batch_count, status;
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

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-nhrs")))
      nhrs = atoi(argv[loop_count + 1]);  
    
    else if (!(cmd_argument.compare("-batch_count"))) 
      batch_count = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (A_row <= 0 || A_col <= 0 || A_row != A_col || batch_count <= 0){
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }

  B_row = A_row;
  B_col = nhrs;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, batch_count);
  
  return status;
}
