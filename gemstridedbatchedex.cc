%%writefile ne.cc
#include <unordered_map>
#include "cublas_GemmStridedBatchedEx_test.h"

template<class T>
GemmStridedBatchedEx<T>::GemmStridedBatchedEx(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, T alpha, T beta, char mode, char algo)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), batch_count(batch_count), alpha(alpha), beta(beta), mode(mode), algo(algo) {}

template<class T>
void GemmStridedBatchedEx<T>::FreeMemory() {
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
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int GemmStridedBatchedEx<T>::GemmStridedBatchedExApiCall() {
  //! Allocating Host Memory for Matrices
   HostMatrixA = new T[batch_count * A_row * A_col];
   HostMatrixB = new T[batch_count * B_row * B_col];
   HostMatrixC = new T[batch_count * C_row * C_col];

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

  if (!HostMatrixC) {
    std::cout << "!!!! Host memory allocation error (matrixC)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  // initialize CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle\n";
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
         std::cout << "!!!! Device memory allocation error (A)\n";
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(float), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         std::cout << "!!!! Device memory allocation error (B)\n";
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(float), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         std::cout << "!!!! Device memory allocation error (C)\n";
         return EXIT_FAILURE;
       }

       break;
     }

     case 'C': {
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
         std::cout << "!!!! Device memory allocation error (A)\n";
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * B_row * B_col, sizeof(float), (void**)&DeviceMatrixB);
       if (status != CUBLAS_STATUS_SUCCESS) {
         std::cout << "!!!! Device memory allocation error (B)\n";
         return EXIT_FAILURE;
       }
       status = cublasAlloc(batch_count * C_row * C_col, sizeof(float), (void**)&DeviceMatrixC);
       if (status != CUBLAS_STATUS_SUCCESS) {
         std::cout << "!!!! Device memory allocation error (C)\n";
         return EXIT_FAILURE;
       }

       break;
     }
  }
   
   //! Setting the values of matrices on device
   cudaStatus = cudaMemcpy(DeviceMatrixA, HostMatrixA, sizeof(T) * batch_count * A_row * A_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     std::cout << "!!!! Setting up values on device for matrix (A) failed\n";
     return EXIT_FAILURE;
   }
   cudaStatus = cudaMemcpy(DeviceMatrixB, HostMatrixB, sizeof(T) * batch_count * B_row * B_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     std::cout << "!!!! Setting up values on device for matrix (B) failed\n";
     return EXIT_FAILURE;
   }
   cudaStatus = cudaMemcpy(DeviceMatrixC, HostMatrixC, sizeof(T) * batch_count * C_row * C_col, cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
     std::cout << "!!!! Setting up values on device for matrix (C) failed\n";
     return EXIT_FAILURE;
   }
   
   //! Defining stride to differentiate between each batch
   long long int strideA = A_row * A_col;
   long long int strideB = B_row * B_col;
   long long int strideC = C_row * C_col;


  //! Algorithm Selection
  if (algo == 'S') {
    cublas_algo = CUBLAS_GEMM_DEFAULT;
    std::cout <<"\nUsing CUBLAS_GEMM_DEFAULT Algorithm\n";
  }
  
  /**
   * API call to performs matrix - matrix multiplication : \f$ C[i] = alpha * A[i] * B[i] + beta * C[i] \f$ \n
   * This function is an extension of cublas<t>gemmStridedBatched that performs the matrix-matrix multiplication of a batch of matrices
     and allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run.
   * Like cublas<t>gemmStridedBatched, the batch is considered to be "uniform", i.e. all instances have the same dimensions (m, n, k), 
     leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. Input matrices A, B and output matrix C for each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance. 
   * Pointers to A, B and C matrices for the first instance are passed to the function by the user along with the offsets in number of 
     elements - strideA, strideB and strideC that determine the locations of input and output matrices in future instances.
   * The second variant of cublasGemmStridedBatchedEx function is provided for backward compatibility with C++ applications code, where 
     the computeType parameter is of cudaDataType_ instead of cublasComputeType_t.
   * C[i] matrices must not overlap, i.e. the individual gemm operations must be computable independently; otherwise, undefined behavior 
     is expected.  
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_ARCH_MISMATCH - CublasGemmBatchedEx is only supported for GPU with architecture capabilities equal or greater than 5.0
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters Atype, Btype and Ctype or the algorithm, algo is not supported.
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n, k, batchCount < 0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SGemmStridedBatchedEx API\n";
      clk_start = clock();
 
      status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (float *)&alpha, 
                                         (float *)DeviceMatrixA, CUDA_R_32F, A_row, strideA,
                                         (float *)DeviceMatrixB, CUDA_R_32F, B_row, strideB,
                                         (float *)&beta, (float *)DeviceMatrixC, CUDA_R_32F,
                                         C_row, strideC, batch_count, CUBLAS_COMPUTE_32F, cublas_algo);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SGemmStridedBatchedEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SGemmStridedBatchedEx API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CGemmStridedBatchedEx API\n";
      clk_start = clock();
       
      status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         A_row, B_col, A_col, (float *)&alpha, 
                                         (float *)DeviceMatrixA, CUDA_R_32F, A_row, strideA,
                                         (float *)DeviceMatrixB, CUDA_R_32F, B_row, strideB,
                                         (float *)&beta, (float *)DeviceMatrixC, CUDA_R_32F,
                                         C_row, strideC, batch_count, CUDA_R_32F, cublas_algo);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CGemmStridedBatchedEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CGemmStridedBatchedEx API call ended\n";
      break;
    }

  }
  
  //! Getting the final output
  cudaStatus = cudaMemcpy(HostMatrixC, DeviceMatrixC,  sizeof(T) * batch_count * C_row * C_col, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::cout << "!!!! Failed to to Get values in Host Matrix C";
    return EXIT_FAILURE;
  }


  std::cout << "\nMatrix C after " << mode << "GemmStridedBatchedEx operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintStridedBatchedMatrix<float>((float *)HostMatrixC, C_row, C_col, batch_count);
      break;
    }

    case 'C': {
      util::PrintStridedBatchedMatrix<float>((float *)HostMatrixC, C_row, C_col, batch_count);
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

int mode_S(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, float alpha, float beta, char algo) {
   
  GemmStridedBatchedEx<float> SGemmStridedBatchedEx(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'S', algo);
  return SGemmStridedBatchedEx.GemmStridedBatchedExApiCall();
}

int mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, float alpha, float beta, char algo) {

  GemmStridedBatchedEx<float> CGemmStridedBatchedEx(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, 'C', algo);
  return CGemmStridedBatchedEx.GemmStridedBatchedExApiCall();
}


int (*cublas_func_ptr[])(int, int, int, int, int, int, int, float, float, char) = {
  mode_S, mode_C, 
};

int main(int argc, char **argv) {

  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count, status;
  float alpha, beta;
  char mode, algo;
  
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['C'] = 1;

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

    else if (!(cmd_argument.compare("-alpha")))
      alpha = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-algo")))
      algo = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (A_row <= 0 || A_col <= 0 || B_col <= 0 || batch_count <= 0) {
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }
  
  //! initializing values for matrix B and C
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, algo);

  return status;
}
