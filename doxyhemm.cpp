#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/**
 * 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 * finally dividing it by latency to get required throughput 
 */
#define THROUGHPUT(clk_start, clk_end, operations) ((1e-9 * 2 * operations) / (clk_end - clk_start)) 

template<class T>
class Hemm {
  private:
    int A_row, A_col, B_row, B_col, C_row, C_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixB;
    T *HostMatrixC;
    T *DeviceMatrixA;
    T *DeviceMatrixB;
    T *DeviceMatrixC;
    T alpha;
    T beta;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    //! Hemm constructor - to initialize the global varibles using initializer list
    Hemm(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
        : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
          C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

    //! FreeMemory function - to free the allocated memory when program is ended or in case of any error
    void FreeMemory(){
      if (HostMatrixA)
        delete[] HostMatrixA;
      
      if (HostMatrixB)
        delete[] HostMatrixB;

      if (HostMatrixC)
        delete[] HostMatrixC;
      
      cudaStatus = cudaFree(DeviceMatrixA);  //!< free device memory for Matrix A
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;   
      }
      
      cudaStatus = cudaFree(DeviceMatrixB);  //!< free device memory for Matrix B
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for B" << std::endl;
      }
      
      cudaStatus = cudaFree(DeviceMatrixC);  //!< free device memory for Matrix C
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for C" << std::endl;
      }

      status  = cublasDestroy(handle);  //!< destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }
   
    int HemmApiCall() {
      //! Host Memory Allocation for Matrices
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
      
      /** define the lower triangle of an mxm Hermitian matrix A in lower mode column by column
       *  define mxn matrix B and C column by column
       *  using RANDOM macro to generate random numbers
       */
      switch (mode) {
        case 'C': {
          util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

          //! printing input matrices
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

          //! printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          std::cout << "\nMatrix B:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
          std::cout << "\nMatrix C:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
          break; 
        }
      }

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

      //! initialize CUBLAS context
      status = cublasCreate(&handle);      
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to initialize handle\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      //! copy matrices from the host to the device
      status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);  //!< A -> d_A
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix A from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);  //!< B -> d_B
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix B from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  //!< C -> d_C
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix C from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      switch (mode) {
        case 'C': {
          std::cout << "\nCalling Chemm API\n";
          clk_start = clock();
          /**
           * calling hemm API based on mode passed 
           * storing result in device Matrix C
           */
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
          /**
           * calling hemm API based on mode passed 
           * storing result in device Matrix C
           */
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

      //! Copying Matrices from device to host
      status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                              DeviceMatrixC, C_row, HostMatrixC, C_row);  //!< copy d_c -> C

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
};        

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

  //! reading cmd line arguments
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
 
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  //! function call
  switch (mode) {
    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
      cuComplex beta = {(float)beta_real, (float)beta_imaginary};

      Hemm<cuComplex> Chemm(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Chemm.HemmApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
      cuDoubleComplex beta = {beta_real, beta_imaginary};

      Hemm<cuDoubleComplex> Zhemm(A_row, A_col,B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Zhemm.HemmApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}
