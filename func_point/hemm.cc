#include "cublas_hemm_test.h"
#include <unordered_map>

template<class T>
Hemm<T>::Hemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Hemm<T>::FreeMemory(){
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
int Hemm<T>::HemmApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col]; 
  HostMatrixB = new T[B_row * B_col]; 
  HostMatrixC = new T[C_row * C_col]; 
  
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
  
  /**
   * Switch Case - To Initialize and Print input matrices based on mode passed
   * A is a Hermitian matrix, B and C are general matrices
   */
  switch (mode) {
    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      break; 
    }
                        
    case 'Z': {
      util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
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

  cudaStatus = cudaMalloc((void **)&DeviceMatrixC, C_row * C_col * sizeof(*HostMatrixC));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
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

  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);  // A -> d_A
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);  // B -> d_B
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  // C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to performs matrix - matrix multiplication : C = alpha * A * B + beta * C
   */
     
  /**
   * The possible error values returned by this API and their meanings are listed below :
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m, n <0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU
   */
  switch (mode) {
    case 'C': {
      std::cout << "\nCalling Chemm API\n";
      clk_start = clock();

      status = cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            A_row, B_col, (cuComplex *)&alpha,
                            (cuComplex *) DeviceMatrixA, A_row, 
                            (cuComplex *)DeviceMatrixB, B_row, (cuComplex *)&beta, 
                            (cuComplex *)DeviceMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Chemm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Chemm API call ended\n";
      break;
    }
  
    case 'Z': {
      std::cout << "\nCalling Zhemm API\n";
      clk_start = clock();

      status = cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            A_row, B_col, (cuDoubleComplex *)&alpha,
                            (cuDoubleComplex *)DeviceMatrixA, A_row, 
                            (cuDoubleComplex *)DeviceMatrixB, B_row, 
                            (cuDoubleComplex *)&beta, (cuDoubleComplex *)DeviceMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zhemm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zhemm API call ended\n";
      break;
    }
  }

  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                          DeviceMatrixC, C_row, HostMatrixC, C_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatriz C after " << mode << "hemm operation is:\n";

  switch (mode) {
    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
      break;
    }
  }

  long long total_operations = A_row * A_col * B_col;  
  
  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";
  
  FreeMemory();
  return EXIT_SUCCESS;      
}

void mode_C(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Hemm<cuComplex> Chemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, 'C');
  Chemm.HemmApiCall();
 
}

void mode_Z(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
  
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Hemm<cuDoubleComplex> Zhemm(A_row, A_col,B_row, B_col, C_row, C_col, alpha, beta, 'Z');
  Zhemm.HemmApiCall(); 
}


void (*cublas_func_ptr[])(int, int, int, int, int, int, double, double, double, double) = {
  mode_C, mode_Z
};


int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, C_row, C_col, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['C'] = 0;
  mode_index['Z'] = 1;

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
 
  //! Initializing values for matrix B and C
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, B_row, B_col, C_row, C_col, alpha_real, alpha_imaginary, beta_real, beta_imaginary);

  return EXIT_SUCCESS;
}
