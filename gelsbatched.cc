

%%writefile cublas_Gelsbatched_test.cc
#include <unordered_map>
#include "cublas_Gelsbatched_test.h"

template<class T>
Gelsbatched<T>::Gelsbatched(int A_row, int A_col, int C_row, int C_col, int batch_count, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), batch_count(batch_count), mode(mode) {}

template<class T>
void Gelsbatched<T>::FreeMemory() {
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
}

template<class T>
int Gelsbatched<T>::GelsbatchedApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T*[batch_count];
   HostMatrixC = new T*[batch_count];
   HostInfoArray = new int [batch_count];
   
   

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
   * A, B and C are general matrices
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);



      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
      

      std::cout << "\nMatrix A:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nMatrix C:\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
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
      std::cout << "!!!! Device memory allocation for matrix (C) failed\n";
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

    status = cublasSetMatrix(C_row, C_col, sizeof(T), HostMatrixC[batch], C_row, HostPtrToDeviceMatC[batch], C_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Matrix C failed\n";
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


   cudaStatus = cudaMalloc((void**)&DeviceInfoArray, batch_count * sizeof(int));
   if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! failed to create Info array\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SGelsbatched API\n";
      clk_start = clock();
 
      status = cublasSgelsBatched(handle, CUBLAS_OP_N, A_row, A_col, C_col, (float **)DeviceMatrixA,
                                  A_row, (float **)DeviceMatrixC, C_row, info, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SGelsbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SGelsbatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DGelsbatched API\n";
      clk_start = clock();

      status = cublasDgelsBatched(handle, CUBLAS_OP_N, A_row, A_col, C_col, (double **)DeviceMatrixA,
                                  A_row, (double **)DeviceMatrixC, C_row, info, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DGelsbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DGelsbatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CGelsbatched API\n";
      clk_start = clock();
       
      status = cublasCgelsBatched(handle, CUBLAS_OP_N, A_row, A_col, C_col, (cuComplex **)DeviceMatrixA,
                                  A_row, (cuComplex **)DeviceMatrixC, C_row, info, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CGelsbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CGelsbatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZGelsbatched API\n";
      clk_start = clock();
   
      status = cublasZgelsBatched(handle, CUBLAS_OP_N, A_row, A_col, C_col, (cuDoubleComplex **)DeviceMatrixA,
                                  A_row, (cuDoubleComplex **)DeviceMatrixC, C_row, info, DeviceInfoArray, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZGelsbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZGelsbatched API call ended\n";
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

    status = cublasGetMatrix(C_row, C_col, sizeof(T), HostPtrToDeviceMatC[batch], 
                             C_row, HostMatrixC[batch], C_row);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! API execution failed\n";
      return EXIT_FAILURE;
    }
  }
  
  

  status = cublasGetVector(batch_count, sizeof (*HostInfoArray),
                           DeviceInfoArray, VECTOR_LEADING_DIMENSION,
		               HostInfoArray, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  } 


  

  std::cout << "\nMatrix A after " << mode << "Gelsbatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::PrintBatchedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
      
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'D': {
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::PrintBatchedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);
      
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'C': {
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
      
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
      std::cout <<"Info status = " <<*info ;
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
      
      util::PrintVector<int>((int *)HostInfoArray, batch_count);
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

int mode_S(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  Gelsbatched<float> SGelsbatched(A_row, A_col, C_row, C_col, batch_count,'S');
  return SGelsbatched.GelsbatchedApiCall();
}

int mode_D(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  Gelsbatched<double> DGelsbatched(A_row, A_col, C_row, C_col, batch_count, 'D');
  return DGelsbatched.GelsbatchedApiCall();
}

int mode_C(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  Gelsbatched<cuComplex> CGelsbatched(A_row, A_col, C_row, C_col, batch_count, 'C');
  return CGelsbatched.GelsbatchedApiCall();

}

int mode_Z(int A_row, int A_col, int C_row, int C_col, int batch_count) {

  Gelsbatched<cuDoubleComplex> ZGelsbatched(A_row, A_col, C_row, C_col, batch_count, 'Z');
  return ZGelsbatched.GelsbatchedApiCall();  
}


int (*cublas_func_ptr[])(int, int, int, int, int) = {
  mode_S,mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, C_row , C_col, batch_count, nhrs, status;
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
    
    else if (!(cmd_argument.compare("-nhrs"))) 
      nhrs = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (A_row <= 0 || A_col <= 0 || nhrs <= 0 || batch_count <= 0){
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }

  C_row = A_col;
  C_col= nhrs;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, batch_count);
  
  return status;
}
