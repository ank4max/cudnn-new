%%writefile cublas_GetriBatched_test.cc


#include <unordered_map>
#include "cublas_GetriBatched_test.h"

template<class T>
GetriBatched<T>::GetriBatched(int A_row, int A_col, int C_row, int C_col, int batch_count, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), batch_count(batch_count), mode(mode) {}

template<class T>
void GetriBatched<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostMatrixC)
    delete[] HostMatrixC;

  if (HostPivotArray)
    delete[] HostPivotArray;
  
  if (HostInfoArray)
    delete[] HostInfoArray;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixC);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
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
int GetriBatched<T>::GetriBatchedApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T*[batch_count];
  HostMatrixC = new T*[batch_count];
  HostPivotArray = new int[A_row * batch_count];
  HostInfoArray = new int[batch_count];
   

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

  if (!HostPivotArray) {
    std::cout << "!!!! Host memory allocation error (Pivotarray)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostInfoArray) {
    std::cout << "!!!! Host memory allocation error (HostInfoArray)\n";
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
      util::BatchedMemoryAllocation<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      
      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::BatchedMemoryAllocation<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::BatchedMemoryAllocation<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
      

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      break;
    }

  
  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];
  HostPtrToDeviceMatC = new T*[batch_count];


  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatC[batch], C_row * C_col * sizeof(T));
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

  cudaStatus = cudaMalloc((void**)&DeviceMatrixC, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for matrix (C) failed\n";
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
  }

  /**
   * API call to perform the inversion of matrices A[i] . \n
   * Prior to calling cublas<t>getriBatched, the matrix A[i] must be factorized first using the routine cublas<t>getrfBatched. 
   * After the call of cublas<t>getrfBatched, the matrix pointing by Aarray[i] will contain the LU factors of the matrix A[i] and 
     the vector pointing by (PivotArray+i) will contain the pivoting sequence.\n
   * Following the LU factorization, cublas<t>getriBatched uses forward and backward triangular solvers to complete inversion of matrices A[i] for i = 0, ..., batchSize-1. 
     The inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of Array[i].
   * This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.
   * If cublas<t>getrfBatched is performed by non-pivoting, PivotArray of cublas<t>getriBatched should be nil.
   */
    
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, batchSize, lda, ldc < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */

  //! calling GetrfBatched API to check for Lu factorization before calling getribatched API 
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SGetrfBatched API\n";
 
      status = cublasSgetrfBatched(handle, A_row, (float**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SGetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      std::cout << "SGetrfBatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DGetrfBatched API\n";

      status = cublasDgetrfBatched(handle, A_row, (double**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DGetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }
      std::cout << "DGetrfBatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CGetrfBatched API\n";
       
      status = cublasCgetrfBatched(handle, A_row, (cuComplex**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CGetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }
      std::cout << "CGetrfBatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZGetrfBatched API\n";
   
      status = cublasZgetrfBatched(handle, A_row, (cuDoubleComplex**)DeviceMatrixA, A_row, 
                                  DevicePivotArray, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZGetrfBatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }


      std::cout << "ZGetrfBatched API call ended\n";
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
  

  status = cublasGetVector(batch_count, sizeof (*HostInfoArray),
                           DeviceInfoArray, VECTOR_LEADING_DIMENSION,
		               HostInfoArray, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  int check = 0;
  //!Checking the value of info variable
  for(batch = 0; batch < batch_count; batch++){
    if(HostInfoArray[batch] != 0) {
      check = 1;
    }
  }

  //! Switch case for Getribatched API
  if(check == 0){
    switch (mode) {
      case 'S': {
        std::cout << "\nCalling SgetriBatched API\n";
        clk_start = clock();
        status = cublasSgetriBatched(handle, A_row, (float**)DeviceMatrixA, A_row, DevicePivotArray, (float**)DeviceMatrixC, 
                                     C_row, DeviceInfoArray, batch_count);

        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cout << "!!!!  SgetriBatched kernel execution error\n";
          FreeMemory();
          return EXIT_FAILURE;
        }
        clk_end = clock();
        std::cout << "SgetriBatched API call ended\n";
        break;
      }

      case 'D': {
        std::cout << "\nCalling DgetrifBatched API\n";
        clk_start = clock();
        status = cublasDgetriBatched(handle, A_row, (double**)DeviceMatrixA, A_row, DevicePivotArray, (double**)DeviceMatrixC, 
                                     C_row, DeviceInfoArray, batch_count);

        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cout << "!!!!  DgetriBatched kernel execution error\n";
          FreeMemory();
          return EXIT_FAILURE;
        }
        clk_end = clock();

        std::cout << "DgetriBatched API call ended\n";
        break;
      }

      case 'C': {
        std::cout << "\nCalling CgetriBatched API\n";
        clk_start = clock();
        status = cublasCgetriBatched(handle, A_row, (cuComplex**)DeviceMatrixA, A_row, DevicePivotArray, (cuComplex**)DeviceMatrixC, 
                                     C_row, DeviceInfoArray, batch_count);

        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cout << "!!!!  CgetriBatched kernel execution error\n";
          FreeMemory();
          return EXIT_FAILURE;
        }

        clk_end = clock();
        std::cout << "CgetriBatched API call ended\n";
        break;
      }

      case 'Z': {
        std::cout << "\nCalling ZgetriBatched API\n";
        clk_start = clock();
        status = cublasZgetriBatched(handle, A_row, (cuDoubleComplex**)DeviceMatrixA, A_row, DevicePivotArray, (cuDoubleComplex**)DeviceMatrixC, 
                                     C_row, DeviceInfoArray, batch_count);

        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cout << "!!!!  ZgetriBatched kernel execution error\n";
          FreeMemory();
          return EXIT_FAILURE;
        }
        clk_end = clock();

        std::cout << "ZGetrfBatched API call ended\n";
        break;
      }
    
    }
  }

  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetMatrix(C_row, C_col, sizeof(T), HostPtrToDeviceMatC[batch], 
                             C_row, HostMatrixC[batch], C_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout <<" Failed to copy C from device to host\n";
      return EXIT_FAILURE;
    }
  }



  switch (mode) {
    case 'S': {
      std::cout << "\nMatrix A after " << mode << "GetrfBatched operation is:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "The pivoting sequence : " << "\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout << "The Info status : " << "\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout << "\nThe Inverse of A  after " << mode << "getriBatched operation is: \n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'D': {
      std::cout << "\nMatrix A after " << mode << "GetrfBatched operation is:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "The pivoting sequence : " << "\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout << "The Info status : " << "\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout << "\nThe Inverse of A  after " << mode << "getriBatched operation is: \n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'C': {
      std::cout << "\nMatrix A after " << mode << "GetrfBatched operation is:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "The pivoting sequence : " << "\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout << "The Info status : " << "\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout << "\nThe Inverse of A  after " << mode << "getriBatched operation is: \n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'Z': {
      std::cout << "\nMatrix A after " << mode << "GetrfBatched operation is:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "The pivoting sequence : " << "\n";
      util::PrintMatrix<int>((int *)HostPivotArray, A_row, batch_count);
      std::cout << "The Info status : " << "\n";
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout << "\nThe Inverse of A  after " << mode << "getriBatched operation is: \n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
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

int mode_S(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  GetriBatched<float> SGetriBatched(A_row, A_col, C_row, C_col, batch_count,'S');
  return SGetriBatched.GetriBatchedApiCall();
}

int mode_D(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  GetriBatched<double> DGetriBatched(A_row, A_col, C_row, C_col, batch_count, 'D');
  return DGetriBatched.GetriBatchedApiCall();
}

int mode_C(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  GetriBatched<cuComplex> CGetriBatched(A_row, A_col, C_row, C_col, batch_count, 'C');
  return CGetriBatched.GetriBatchedApiCall();

}

int mode_Z(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  GetriBatched<cuDoubleComplex> ZGetriBatched(A_row, A_col, C_row, C_col, batch_count, 'Z');
  return ZGetriBatched.GetriBatchedApiCall();  
}


int (*cublas_func_ptr[])(int, int, int, int, int) = {
  mode_S,mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, C_row, C_col, batch_count, status;
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
  if (A_row <= 0 || A_col <= 0 || A_row != A_col || batch_count <= 0){
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }
  C_row = A_row;
  C_col = A_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, batch_count);
  
  return status;
}
