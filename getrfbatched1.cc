%%writefile cublas_GetrfBatched_test.cc


#include <unordered_map>
#include "cublas_GetrfBatched_test.h"

template<class T>
GetrfBatched<T>::GetrfBatched(int A_row, int A_col, int batch_count, char mode)
    : A_row(A_row), A_col(A_col), batch_count(batch_count), mode(mode) {}

template<class T>
void GetrfBatched<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostPivotArray)
    delete[] HostPivotArray;
  
  if (HostInfoArray)
    delete[] HostInfoArray;


  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DevicePivotArray);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Pivot array" << std::endl;
  }

  cudaStatus = cudaFree(DeviceInfoArray);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Info array" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GetrfBatched<T>::GetrfBatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostPivotArray = new int[A_row * batch_count];
   HostInfoArray = new int[batch_count];
   

   if (!HostMatrixA) {
     std::cout << "!!!! Host memory allocation error (matrixA)\n";
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
      
      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);


      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];

  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
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

  cudaStatus = cudaMalloc((void**)&DevicePivotArray, A_row * batch_count * sizeof(int));
   if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! failed to create pivot array\n";
    FreeMemory();
    return EXIT_FAILURE;
  }


   cudaStatus = cudaMalloc((void**)&DeviceInfoArray, batch_count * sizeof(int));
   if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! failed to create Info array\n";
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
  
  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasSetMatrix(A_row, A_col, sizeof(T), HostMatrixA[batch], A_row, HostPtrToDeviceMatA[batch], A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix A failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
  }

  /**
   * API call to perform the LU factorization of each Aarray[i] : \f$ P * Aarray[i] = L * U \f$ \n
   * where P is a permutation matrix which represents partial pivoting with row interchanges.\n 
   * L is a lower triangular matrix with unit diagonal and U is an upper triangular matrix.\n
   * L and U are written back to original matrix A, and diagonal elements of L are discarded. \n
   * This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.\n
   * GetrfBatched supports non-pivot LU factorization if PivotArray is nil.\n
   * GetrfBatched supports arbitrary dimension.\n
   * GetrfBatched only supports compute capability 2.0 or above.\n
   */
    
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, batchSize, lda < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */

  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SgetrfBatched API\n";
      clk_start = clock();
 
      status = cublasSgetrfBatched(handle, A_row, (float**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SgetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SgetrfBatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DgetrfBatched API\n";
      clk_start = clock();

      status = cublasDgetrfBatched(handle, A_row, (double**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DgetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DgetrfBatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CgetrfBatched API\n";
      clk_start = clock();
       
      status = cublasCgetrfBatched(handle, A_row, (cuComplex**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CgetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CgetrfBatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZgetrfBatched API\n";
      clk_start = clock();
   
      status = cublasZgetrfBatched(handle, A_row, (cuDoubleComplex**)DeviceMatrixA, A_row, 
                                   DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZgetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZgetrfBatched API call ended\n";
      break;
    }
    
  }
  
  //! Copy Matrix A, holding resultant matrix, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetMatrix(A_row, A_col, sizeof(T), HostPtrToDeviceMatA[batch], 
                             A_row, HostMatrixA[batch], A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }
  }
  
  status = cublasGetMatrix(A_row, batch_count, sizeof(*HostPivotArray), DevicePivotArray, 
                             A_row, HostPivotArray, A_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }
  
  //! Copying Info array from device to host
  status = cublasGetVector(batch_count, sizeof (*HostInfoArray),
                           DeviceInfoArray, VECTOR_LEADING_DIMENSION,
		               HostInfoArray, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  } 



  std::cout << "\nMatrix A after " << mode << "GetrfBatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout <<"The Pivoting sequence is : " <<"\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout <<"Info array : " <<"\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      break;
    }

    case 'D': {
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout <<"The Pivoting sequence is : " <<"\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout <<"Info array : " <<"\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      break;
    }

    case 'C': {
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout <<"The Pivoting sequence is : " <<"\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout <<"Info array : " <<"\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout <<"The Pivoting sequence is : " <<"\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout <<"Info array : " <<"\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
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

int mode_S(int A_row, int A_col, int batch_count) {

  GetrfBatched<float> SGetrfBatched(A_row, A_col, batch_count,'S');
  return SGetrfBatched.GetrfBatchedApiCall();
}

int mode_D(int A_row, int A_col, int batch_count) {

  GetrfBatched<double> DGetrfBatched(A_row, A_col, batch_count, 'D');
  return DGetrfBatched.GetrfBatchedApiCall();
}

int mode_C(int A_row, int A_col, int batch_count) {

  GetrfBatched<cuComplex> CGetrfBatched(A_row, A_col, batch_count, 'C');
  return CGetrfBatched.GetrfBatchedApiCall();

}

int mode_Z(int A_row, int A_col, int batch_count) {

  GetrfBatched<cuDoubleComplex> ZGetrfBatched(A_row, A_col, batch_count, 'Z');
  return ZGetrfBatched.GetrfBatchedApiCall();  
}


int (*cublas_func_ptr[])(int, int, int) = {
  mode_S,mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, batch_count, status;
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

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, batch_count);
  
  return status;
}
