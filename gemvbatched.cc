%%writefile gemv.cc
#include <unordered_map>
#include "gemv.h"

template<class T>
GemvBatched<T>::GemvBatched(int A_row, int A_col, int X_length, int Y_length, int batch_count, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), X_length(X_length), Y_length(Y_length), batch_count(batch_count),
      alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void GemvBatched<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostVectorX)
    delete[] HostVectorX;

  if (HostVectorY)
    delete[] HostVectorY;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorX);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorY);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Y" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GemvBatched<T>::GemvBatchedApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T *[batch_count];
  HostVectorX = new T *[batch_count];
  HostVectorY = new T *[batch_count];

  if (!HostMatrixA) {
    std::cout << "!!!! Host memory allocation error (matrixA)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    std::cout << "!!!! Host memory allocation error (vectorY)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A is a general matrix, X and Y are vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<float>((float **)HostVectorX, X_length, batch_count);
      util::InitializeBatchedVector<float>((float **)HostVectorY, Y_length, batch_count);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintBatchedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintBatchedVector<float>((float **)HostVectorX, X_length, batch_count);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintBatchedVector<float>((float **)HostVectorY, Y_length, batch_count);
      break;
    }

    case 'D': {
      util::InitializeBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<double>((double **)HostVectorX, X_length, batch_count);
      util::InitializeBatchedVector<double>((double **)HostVectorY, Y_length, batch_count);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintBatchedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintBatchedVector<double>((double **)HostVectorX, X_length, batch_count);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintBatchedVector<double>((double **)HostVectorY, Y_length, batch_count);
       
      break;
    }

    case 'C': {
      util::InitializeBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexVector<cuComplex>((cuComplex **)HostVectorX, X_length, batch_count);
      util::InitializeBatchedComplexVector<cuComplex>((cuComplex **)HostVectorX, X_length, batch_count);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintBatchedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintBatchedComplexVector<cuComplex>((cuComplex **)HostVectorX, X_length, batch_count);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintBatchedComplexVector<cuComplex>((cuComplex **)HostVectorY, Y_length, batch_count);
      
      break;
    }

    case 'Z': {
      util::InitializeBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostVectorX, X_length, batch_count);
      util::InitializeBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostVectorX, X_length, batch_count);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintBatchedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostVectorX, X_length, batch_count);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostVectorY, Y_length, batch_count);     
      
      break;
    }

    case 'H': {
      util::InitializeBatchedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
      util::InitializeBatchedVector<__half>((__half **)HostVectorX, X_length, batch_count);
      util::InitializeBatchedVector<__half>((__half **)HostVectorY, Y_length, batch_count);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintBatchedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintBatchedVector<__half>((__half **)HostVectorX, X_length, batch_count);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintBatchedVector<__half>((__half **)HostVectorY, Y_length, batch_count);
       
      break;
    }

  }
  
  //! Allocating matrices on device    
  HostPtrToDeviceMatA = new T*[batch_count];
  HostPtrToDeviceVectorX = new T*[batch_count];
  HostPtrToDeviceVectorY = new T*[batch_count];

  int batch;

  for(batch = 0; batch < batch_count; batch++) {
    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for matrix (A) failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceVectorX[batch], X_length * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for vector X failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc((void**)&HostPtrToDeviceVectorY[batch], Y_length * sizeof(T));
    if (cudaStatus != cudaSuccess) {
      std::cout << "!!!! Device memory allocation for vector y failed\n";
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

  cudaStatus = cudaMalloc((void**)&DeviceVectorX, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for vector X failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void**)&DeviceVectorY, batch_count * sizeof(T*));
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Device memory allocation for vector Y failed\n";
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
  cudaStatus = cudaMemcpy(DeviceVectorX, HostPtrToDeviceVectorX, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Memory copy on device for matrix (B) failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceVectorY, HostPtrToDeviceVectorY, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
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

    status = cublasSetVector(X_length, sizeof(T), HostVectorX[batch], 
                           VECTOR_LEADING_DIMENSION, HostPtrToDeviceVectorX[batch], VECTOR_LEADING_DIMENSION);

    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for vector X failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
    
    status = cublasSetVector(Y_length, sizeof(T), HostVectorY[batch], 
                           VECTOR_LEADING_DIMENSION, HostPtrToDeviceVectorY[batch], VECTOR_LEADING_DIMENSION);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "!!!! Setting up values on device for Vector Y failed\n";
      FreeMemory();
      return EXIT_FAILURE;
    }
  }
  
  
  

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - the parameters m, n < 0 or incx, incy = 0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to performs matrix - vector multiplication : \f$ Y = alpha * A * X + beta * Y \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sgemvbatched API\n";
      clk_start = clock();

      status = cublasSgemvBatched(handle, CUBLAS_OP_N, A_row, A_col, (float *)alpha,
                                  (float**)DeviceMatrixA, A_row, (float**)DeviceVectorX, 
                                   VECTOR_LEADING_DIMENSION, (float *)&beta, (float **)DeviceVectorY, 
                                   VECTOR_LEADING_DIMENSION, batch_count);
                                

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Sgemvbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sgemvbatched API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dgemvbatched API\n";
      clk_start = clock();

      status = cublasDgemvBatched(handle, CUBLAS_OP_N, A_row, A_col, (double *)alpha,
                                  (double **)DeviceMatrixA, A_row, (double **)DeviceVectorX, 
                                   VECTOR_LEADING_DIMENSION, (double *)&beta, (double **)DeviceVectorY, 
                                   VECTOR_LEADING_DIMENSION, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dgemvbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dgemvbatched API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cgemvbatched API\n";
      clk_start = clock();

      status = cublasCgemvbatched(handle, CUBLAS_OP_N, A_row, A_col, (cuComplex *)alpha,
                                  (cuComplex **)DeviceMatrixA, A_row, (cuComplex **)DeviceVectorX, 
                                   VECTOR_LEADING_DIMENSION, (cuComplex *)&beta, (cuComplex **)DeviceVectorY, 
                                   VECTOR_LEADING_DIMENSION, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Cgemvbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgemvbatched API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zgemvbatched API\n";
      clk_start = clock();

      status = cublasZgemvbatched(handle, CUBLAS_OP_N, A_row, A_col, (cuDoubleComplex *)alpha,
                                  (cuDoubleComplex **)DeviceMatrixA, A_row, (cuDoubleComplex **)DeviceVectorX, 
                                   VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)&beta, (cuDoubleComplex **)DeviceVectorY, 
                                   VECTOR_LEADING_DIMENSION, batch_count);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zgemvbatched kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zgemvbatched API call ended\n";
      break;
    }
  }

  //! Copy Vector Y, holding resultant vector, from Device to Host using cublasGetMatrix()
  //! getting the final output
  for (batch = 0; batch < batch_count; batch++) {
    status = cublasGetVector(Y_length, sizeof(T), HostPtrToDeviceVectorY[batch], 
                             VECTOR_LEADING_DIMENSION, HostVectorY[batch], VECTOR_LEADING_DIMENSION);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "Unable to get output vector Y from device\n";
      return EXIT_FAILURE;
    }
  }
  

  std::cout << "\nVector Y after " << mode << "gemvbatched operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintBatchedVector<float>((float **)HostVectorY, Y_length, batch_count);
      break;
    }

    case 'D': {
      util::PrintBatchedVector<double>((double **)HostVectorY, Y_length, batch_count);
      break;
    }

    case 'C': {
      util::PrintBatchedComplexVector<cuComplex>((cuComplex **)HostVectorY, Y_length, batch_count);
      break;
    }

    case 'Z': {
      util::PrintBatchedComplexVector<cuDoubleComplex>((cuDoubleComplex **)HostVectorY, Y_length, batch_count);
      break;
    }

    case 'H': {
      util::PrintBatchedComplexVector<__half>((__half **)HostVectorY, Y_length, batch_count);
      break;
    }
  }

  long long total_operations = A_row * X_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col, int X_length, int Y_length, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  GemvBatched<float> Sgemvbatched(A_row, A_col, X_length, Y_length, batch_count, alpha, beta, 'S');
  return Sgemvbatched.GemvBatchedApiCall();
}

int mode_D(int A_row, int A_col, int X_length, int Y_length, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {   
  double alpha = alpha_real;
  double beta = beta_real;

  GemvBatched<double> Dgemvbatched(A_row, A_col, X_length, Y_length, batch_count, alpha, beta, 'D');
  return Dgemvbatched.GemvBatchedApiCall();
}

int mode_C(int A_row, int A_col, int X_length, int Y_length, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  GemvBatched<cuComplex> Cgemvbatched(A_row, A_col, X_length, Y_length, batch_count, alpha, beta, 'C');
  return Cgemvbatched.GemvBatchedApiCall(); 
}

int mode_Z(int A_row, int A_col, int X_length, int Y_length, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  GemvBatched<cuDoubleComplex> Zgemvbatched(A_row, A_col, X_length, Y_length, batch_count, alpha, beta, 'Z');
  return Zgemvbatched.GemvBatchedApiCall(); 
}

int mode_H(int A_row, int A_col, int X_length, int Y_length, int batch_count, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {   
   __half alpha = alpha_real;
   __half beta = beta_real;

  GemvBatched< __half> Hgemvbatched(A_row, A_col, X_length, Y_length, batch_count, alpha, beta, 'H');
  return Hgemvbatched.GemvBatchedApiCall();
}



int (*cublas_func_ptr[])(int, int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z, mode_H
};

int main(int argc, char **argv) {

  int A_row, A_col, X_length, Y_length, status, batch_count;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['Z'] = 3;
  mode_index['H'] = 4;

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
  if (A_row <= 0 || A_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  //! initializing values for matrix B and C
  X_length = A_col;
  Y_length = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, X_length, Y_length, batch_count, alpha_real, 
                                       alpha_imaginary, beta_real, beta_imaginary);
  
  return status;
}
