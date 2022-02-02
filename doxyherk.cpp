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
class Herk {
  private:
    int A_row, A_col, C_row, C_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixC;
    T *DeviceMatrixA;
    T *DeviceMatrixC;
    double alpha;
    double beta;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    //! Herk constructor - to initialize the global varibles using initializer list
    Herk(int A_row, int A_col, int C_row, int C_col, double alpha, double beta, char mode)
        : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col),
          alpha(alpha), beta(beta), mode(mode) {}

    void FreeMemory(){
      if (HostMatrixA)
        delete[] HostMatrixA;

      if (HostMatrixC)
        delete[] HostMatrixC;
      
      cudaStatus = cudaFree(DeviceMatrixA);  //!< free device memory A
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;   
      }
      
      cudaStatus = cudaFree(DeviceMatrixC);  //!< free device memory C
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for C" << std::endl;
      }

      status  = cublasDestroy(handle);  //!< destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }

    int HerkApiCall() {
      //! Host Memory Allocation for Matrices
      HostMatrixA = new T[A_row * A_col];
      HostMatrixC = new T[C_row * C_col]; 
      
      if (!HostMatrixA) {
        fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      if (!HostMatrixC) {
        fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      /**
       * define  matrices A column by column
       * define the lower triangle of an nxn Hermitian matrix C
       * using RANDOM macro to generate random numbers
       */
      switch (mode) {
        case 'C': {
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

          //! printing input matrices
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          break; 
        }
                            
        case 'Z': {
          util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

          //! printing input matrices
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          break; 
        }
      }

      cudaStatus = cudaMalloc((void **)&DeviceMatrixA, A_row * A_col * sizeof(*HostMatrixA));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for A " << std::endl;
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

      status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  //!< C -> d_C
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix C from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      switch (mode) {
        case 'C': {
          float alpha_f = (float)alpha;
          float beta_f = (float)beta;  
          std::cout << "\nCalling Cherk API\n";
          clk_start = clock();
          /**
           * calling herk API based on mode passed 
           * storing result in device Matrix C
           */
          status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, &alpha_f, 
                                (cuComplex *)DeviceMatrixA, A_row,
                                &beta_f, (cuComplex *)DeviceMatrixC, C_row); 

        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Cherk kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Cherk API call ended\n";
          break;
        }
      
        case 'Z': {
          std::cout << "\nCalling Zherk API\n";
          clk_start = clock();
          /**
           * calling herk API based on mode passed 
           * storing result in device Matrix C
           */
          status = cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, &alpha,
                                (cuDoubleComplex *)DeviceMatrixA, A_row,
                                &beta, (cuDoubleComplex *)DeviceMatrixC, C_row); 
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Zherk kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Zherk API call ended\n";
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
      
      std::cout << "\nMatriz C after " << mode << "herk operation is:\n";

      switch (mode) {
        case 'C': {
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col); 
          break;
        }

        case 'Z': {
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col); 
          break;
        }
      }

      long long total_operations = A_row * A_col * C_col;
      //! printing latency and throughput of the function
      std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
                   "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";
      
      FreeMemory();

      return EXIT_SUCCESS;     
    }
};        


int main(int argc, char **argv) {
  int A_row, A_col, C_row, C_col, status;
  double alpha_real, beta_real;
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
      
    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  //! initializing values for matrix C
  C_row = A_row;
  C_col = A_row;

  //! function call
  switch (mode) {
    case 'C': {
      Herk<cuComplex> Cherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, mode);
      status = Cherk.HerkApiCall();
      break;
    }

    case 'Z': {
      Herk<cuDoubleComplex> Zherk(A_row, A_col, C_row, C_col, alpha_real, beta_real, mode);
      status = Zherk.HerkApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}
