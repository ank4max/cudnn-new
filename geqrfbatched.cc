

%%writefile cublas_GeqrfBatched_test.cc
#include <unordered_map>
#include "cublas_GeqrfBatched_test.h"

template<class T>
GeqrfBatched<T>::GeqrfBatched(int A_row, int A_col, int vector_length, int batch_count, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length), batch_count(batch_count), mode(mode) {}

template<class T>
void GeqrfBatched<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;
  
  if (HostTauArray)
    delete[] HostTauArray;


  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceTauArray);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for TauArray" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GeqrfBatched<T>::GeqrfBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostTauArray = new T*[batch_count];
  

   if (!HostMatrixA) {
     std::cout << "!!!! Host memory allocation error (matrixA)\n";
     FreeMemory();
     return EXIT_FAILURE;
   }

   if (!HostTauArray) {
     std::cout << "!!!! Host memory allocation error (TauArray)\n";
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
      util::InitializeBatchedVector<float>((float **)HostTauArray, vector_length, batch_count);
      
      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<double>((double **)HostTauArray, vector_length, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<cuComplex>((cuComplex **)HostTauArray, vector_length, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<cuComplex>((cuComplex **)HostTauArray, vector_length, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

  
  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];
  HostPtrToDeviceTauArray = new T*[batch_count];


  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceTauArray[batch], vector_length * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for TauArray failed\n";
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

  cudaStatus = cudaMalloc((void**)&DeviceTauArray, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for TauArray failed\n";
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

  cudaStatus = cudaMemcpy(DeviceTauArray, HostPtrToDeviceTauArray, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
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

    status = cublasSetVector(vector_length, sizeof(T), HostTauArray[batch], VECTOR_LEADING_DIMENSION, HostPtrToDeviceTauArray[batch], VECTOR_LEADING_DIMENSION);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for TauArray failed\n";
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

   int *info;
   info = new int;


  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SGeqrfBatched API\n";
      clk_start = clock();
 
      status = cublasSgeqrfBatched(handle, A_row, A_col, (float **)DeviceMatrixA,
                                  A_row, (float **)DeviceTauArray,info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SGeqrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SGeqrfBatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DGeqrfBatched API\n";
      clk_start = clock();

      status = cublasDgeqrfBatched(handle, A_row, A_col, (double **)DeviceMatrixA,
                                  A_row, (double **)DeviceTauArray, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DGeqrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DGeqrfBatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CGeqrfBatched API\n";
      clk_start = clock();
       
      status = cublasCgeqrfBatched(handle, A_row, A_col, (cuComplex **)DeviceMatrixA,
                                  A_row, (cuComplex **)DeviceTauArray, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CGeqrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CGeqrfBatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZGeqrfBatched API\n";
      clk_start = clock();
   
      status = cublasZgeqrfBatched(handle, A_row, A_col, (cuDoubleComplex **)DeviceMatrixA,
                                  A_row, (cuDoubleComplex **)DeviceTauArray, info, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZGeqrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZGeqrfBatched API call ended\n";
      break;
    }
    
  }
  
  //! Copy Matrix A, holding resultant matrix, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetVector(vector_length, sizeof(T), HostPtrToDeviceTauArray[batch], 
                             VECTOR_LEADING_DIMENSION, HostTauArray[batch], VECTOR_LEADING_DIMENSION);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }

  }

  std::cout << "\nTauArray after " << mode << "GeqrfBatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedVector<float>((float **)HostTauArray, vector_length, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'D': {
      util::PrintBatchedVector<double>((double **)HostTauArray, vector_length, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'C': {
      util::PrintBatchedComplexVector<cuComplex>((cuComplex **)HostTauArray, vector_length, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostTauArray, vector_length, batch_count);
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

int mode_S(int A_row, int A_col, int vector_length, int batch_count) {

  GeqrfBatched<float> SGeqrfBatched(A_row, A_col, vector_length, batch_count,'S');
  return SGeqrfBatched.GeqrfBatchedApiCall();
}

int mode_D(int A_row, int A_col, int vector_length, int batch_count) {

  GeqrfBatched<double> DGeqrfBatched(A_row, A_col, vector_length, batch_count, 'D');
  return DGeqrfBatched.GeqrfBatchedApiCall();
}

int mode_C(int A_row, int A_col, int vector_length, int batch_count) {

  GeqrfBatched<cuComplex> CGeqrfBatched(A_row, A_col, vector_length, batch_count, 'C');
  return CGeqrfBatched.GeqrfBatchedApiCall();

}

int mode_Z(int A_row, int A_col, int vector_length, int batch_count) {

  GeqrfBatched<cuDoubleComplex> ZGeqrfBatched(A_row, A_col, vector_length, batch_count, 'Z');
  return ZGeqrfBatched.GeqrfBatchedApiCall();  
}


int (*cublas_func_ptr[])(int, int, int, int) = {
  mode_S,mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, batch_count, status;
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
    
    else if (!(cmd_argument.compare("-batch_count"))) 
      batch_count = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (A_row <= 0 || A_col <= 0 || batch_count <= 0){
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }

  vector_length = std::max(1,(std::min(A_row, A_col)));

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, batch_count);
  
  return status;
}
