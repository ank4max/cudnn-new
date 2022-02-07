#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"
#include <trmm.h>


/**
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 * finally dividing it by latency to get required throughput 
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

Trmm:Trmm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, char mode)
    : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
      C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

//! FreeMemory function - To free the allocated memory when program is ended or in case of any error
void Trmm:FreeMemory(){
  if (HostMatrixA)
    delete[] HostMatrixA;
  
  if (HostMatrixB)
    delete[] HostMatrixB;

  if (HostMatrixC)
    delete[] HostMatrixC;
  
  cudaStatus = cudaFree(DeviceMatrixA);  //!< Free device memory for A
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;   
  }
  
  cudaStatus = cudaFree(DeviceMatrixB);  //!< Free device memory for B
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for B" << std::endl;
  }
  
  cudaStatus = cudaFree(DeviceMatrixC);  //!< Free device memory for C
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
  }

  status  = cublasDestroy(handle);  //!< Destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

/**
 * The TrmmAPICall function where host and device memory allocations are done,
      Matrices are set up and a particular variation of Trmm API is called to 
              perform required operation based on the mode passed
 */
int Trmm:TrmmApiCall() {
  //! Host Memory Allocation for Matrices based on dimensions initialized by Trmm constructor
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
   * Switch case to initialize input matrices based on mode passed
   * A is a symmetric Matrix
   * B and C are Non-symmetric matrices and with similar mxn Dimensions
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<float>((float *)HostMatrixC, C_row, C_col);

      //! Printing input matrices
      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      
      break;
    }

    case 'D': {
      util::InitializeSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeMatrix<double>((double *)HostMatrixB, B_row, B_col);
      util::InitializeMatrix<double>((double *)HostMatrixC, C_row, C_col);

      //! Printing input matrices
      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);
      break;  
    }

    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

      //! Printing input matrices
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

      //! Printing input matrices
      std::cout << "\nMatrix A:\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
      break; 
    }
  }
  
  //! Device memory allocations for input matrices 
  //! Required memory is being allocated to device matrices using cudaMalloc() 
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

  //! Initialize CUBLAS context
  status = cublasCreate(&handle);      
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying host matrices values to the device matrices using cublasSetMatrix
      
  /**
   * The function SetMatrix copies a tile of A_row x A_col elements from a matrix A in host to matrix A in 
        device
   */       
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
      
  /**
   * The function SetMatrix copies a tile of B_row x B_col elements from a matrix B in host to matrix B in 
        device
   */
  status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * The function SetMatrix copies a tile of C_row x C_col elements from a matrix C in host to matrix C in 
        device
   */
  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Strmm API\n";
      clk_start = clock();
      
      /**
       * This API performs triangular matrix - matrix multiplication : DeviceMatrixC = alpha * DeviceMatrixA * DeviceMatrixB 
       * DeviceMatrixA - m x m triangular matrix in lower mode ,
       * DeviceMatrixB, DeviceMatrixC - m x n general matrices and alpha - scalar
       */
      status = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B_row, B_col, 
                           (float *)&alpha, (float *)DeviceMatrixA, A_row, 
                           (float *)DeviceMatrixB, B_row, (float *)DeviceMatrixC, C_row);

    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Strmm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Strmm API call ended\n";
      break;
    }
                        
    case 'D': {
      std::cout << "\nCalling Dtrmm API\n";
      clk_start = clock();
      
      /**
       * This API performs triangular matrix - matrix multiplication : DeviceMatrixC = alpha * DeviceMatrixA * DeviceMatrixB 
       * DeviceMatrixA - m x m triangular matrix in lower mode ,
       * DeviceMatrixB, DeviceMatrixC - m x n general matrices and alpha - scalar
       */
      status = cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B_row, B_col, 
                           (double *)&alpha, (double *)DeviceMatrixA, A_row, 
                           (double *)DeviceMatrixB, B_row, (double *)DeviceMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dtrmm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtrmm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctrmm API\n";
      clk_start = clock();
      
      /**
       * This API performs triangular matrix - matrix multiplication : DeviceMatrixC = alpha * DeviceMatrixA * DeviceMatrixB 
       * DeviceMatrixA - m x m triangular matrix in lower mode ,
       * DeviceMatrixB, DeviceMatrixC - m x n general matrices and alpha - scalar
       */
      status = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B_row, B_col, 
                           (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceMatrixB, B_row,
                           (cuComplex *)DeviceMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ctrmm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctrmm API call ended\n";
      break;
    }
  
    case 'Z': {
      std::cout << "\nCalling Ztrmm API\n";
      clk_start = clock();
      
      /**
       * This API performs triangular matrix - matrix multiplication : DeviceMatrixC = alpha * DeviceMatrixA * DeviceMatrixB 
       * DeviceMatrixA - m x m triangular matrix in lower mode ,
       * DeviceMatrixB, DeviceMatrixC - m x n general matrices and alpha - scalar
       */
      status = cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B_row, B_col, 
                           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA,
                           A_row, (cuDoubleComplex *)DeviceMatrixB, B_row,
                           (cuDoubleComplex *)DeviceMatrixC, C_row);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ztrmm kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztrmm API call ended\n";
      break;
    }
  }

  //! Copying Matrix C from device to host using cublasGetMatrix function
  
  /**
   * GetMatrix copies a tile of C_row x C_col from  matrix C in GPU memory space to a matrix C
        in Host Memory space where each element will require (sizeof(*HostMatrixC)) bytes
   */
  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                           DeviceMatrixC, C_row, HostMatrixC, C_row);  //!< copy d_c -> C

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  std::cout << "\nMatrix C after " << mode << "trmm operation is:\n";
  
  //! Printing the final output Matrix C
  switch (mode) {
    case 'S': {
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);  
      break;
    }

    case 'D': {
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);   
      break;
    }

    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col); 
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);  
      break;
    }
  }

  long long total_operations = A_row * A_col * B_col;
  //! Printing latency and throughput of the function
  //! Latency and throughput calculated through time variables used to store API execution time
  
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";
  
  FreeMemory();

  return EXIT_SUCCESS; 
}  


int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, C_row, C_col, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
  char mode;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if (loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  //! Reading cmd line arguments and initializing the required parameters
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

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  //! Initializing input matrices dimensions
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  //! Switch block has cases in which each case will make call to the function based on mode
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;
      float beta = (float)beta_real;

      Trmm<float> Strmm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, mode);
      status = Strmm.TrmmApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;
      double beta = beta_real;

      Trmm<double> Dtrmm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, mode);
      status = Dtrmm.TrmmApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

      Trmm<cuComplex> Ctrmm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, mode);
      status = Ctrmm.TrmmApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

      Trmm<cuDoubleComplex> Ztrmm(A_row, A_col,B_row, B_col, C_row, C_col, alpha, mode);
      status = Ztrmm.TrmmApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}
