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

/**
 * template class Herkx is defined having matrices ,their dimensions,
      mode and scalars quantity declared as private members
 * cublas handle, cuda status and cublas status are also declared as private members
 * clock varibles clk_start and clk_end are to compute throughput and latency
 */
template<class T>
class Herkx {
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
    double beta;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    /**
     * Herkx constructor - to initialize the global varibles using initializer list
     * Herkx constructor initializes the dimensions of input matrices ,the value of
          scalars alpha,beta and sets up the mode of API.
     */
    Herkx(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, double beta, char mode)
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
      
      cudaStatus = cudaFree(DeviceMatrixA);  //!< free device memory for A
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;   
      }
      
      cudaStatus = cudaFree(DeviceMatrixB);  //!< free device memory for B
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for B" << std::endl;
      }
      
      cudaStatus = cudaFree(DeviceMatrixC);  //!< free device memory for C
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for C" << std::endl;
      }

      status  = cublasDestroy(handle);  //!< destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }
  
    /**
     * The HerkxAPICall function where host and device memory allocations are done,
          Matrices are set up and a particular variation of herkx API is called to 
                  perform required operation based on the mode passed
     */
    int HerkxApiCall() {
      //! Host Memory Allocation for Matrices based on dimensions initialized by Herkx constructor
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
       * A and B are non-symmetric Matrices with dimensions nxk
       * C is a Hermitian matrix stored in lower mode
       */
      switch (mode) {
        case 'C': {
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

          //! printing input matrices
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          std::cout << "\nMatrix B:\n";
          util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
          break; 
        }
                            
        case 'Z': {
          util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

          //! printing input matrices
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          std::cout << "\nMatrix B:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
          break; 
        }
      }
      
      //! Device memory allocations for input matrices 
      //! required memory is being allocated to device matrices using cudaMalloc()
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
      
      /**
       * The function SetMatrix copies a tile of A_row x A_col elements from a matrix A in host to matrix A in device
       */   
      status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);  //!< A -> d_A
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix A from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      /**
       * The function SetMatrix copies a tile of B_row x B_col elements from a matrix B in host to matrix B in device
       */
      status = cublasSetMatrix(B_row, B_col, sizeof(*HostMatrixB), HostMatrixB, B_row, DeviceMatrixB, B_row);  //!< B -> d_B
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix B from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      /**
       * The function SetMatrix copies a tile of C_row x C_col elements from a matrix C in host to matrix C in device
       */
      status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  //!< C -> d_C
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix C from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      switch (mode) {
        case 'C': {
          std::cout << "\nCalling Cherkx API\n";
          clk_start = clock();
          
          /**
           * This API performs the Hermitian rank- k update 
           * d_c = alpha*d_a *d_b ^H + beta *d_c
           * d_c - nxn hermitian matrix , d_a and d_b are nxk general matrices 
           * alpha and beta are scalars
           */
          status = cublasCherkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (cuComplex *)&alpha, 
                                (cuComplex *)DeviceMatrixA, A_row,
                                (cuComplex *)DeviceMatrixB, B_row, 
                                (float *)&beta, (cuComplex *)DeviceMatrixC, C_row); 

        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Cherkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Cherkx API call ended\n";
          break;
        }
      
        case 'Z': {
          std::cout << "\nCalling Zherkx API\n";
          clk_start = clock();
          
          /**
           * This API performs the Hermitian rank- k update 
           * d_c = alpha*d_a *d_b ^H + beta *d_c
           * d_c - nxn hermitian matrix , d_a and d_b are nxk general matrices 
           * alpha and beta are scalars
           */
          status = cublasZherkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (cuDoubleComplex *)&alpha, 
                                (cuDoubleComplex *)DeviceMatrixA, A_row,
                                (cuDoubleComplex *)DeviceMatrixB, B_row, 
                                (double *)&beta, (cuDoubleComplex *)DeviceMatrixC, C_row); 
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Zherkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Zherkx API call ended\n";
          break;
        }
      }

      //! Copying Matrices from device to host
      
      /**
       * GetMatrix function copies a tile of C_row x C_col from  matrix C in GPU memory space to a matrix C
            in Host Memory space where each element will require (sizeof(*HostMatrixC)) bytes
       */
      status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                               DeviceMatrixC, C_row, HostMatrixC, C_row);  //!< copy d_c -> C

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      std::cout << "\nMatriz C after " << mode << "herkx operation is:\n";
      
      //! Printing the final output Matrix C
      switch (mode) {
        case 'C': {
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row , C_col); 
          break;
        }

        case 'Z': {
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row , C_col); 
          break;
        }
      }

      long long total_operations = A_row * A_col * B_col;
      
      //! printing latency and throughput of the function
      //! Latency and throughput calculated through time variables used to store API execution time
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

  //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  //! initializing values for matrix B and C        
  B_row = A_row;
  B_col = A_col;
  C_row = A_row;
  C_col = A_row;

  //! Switch block has cases in which any of the cases will be executed to make call to the function based on mode
  switch (mode) {
    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

      Herkx<cuComplex> Cherkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta_real, mode);
      status = Cherkx.HerkxApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

      Herkx<cuDoubleComplex> Zherkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta_real, mode);
      status = Zherkx.HerkxApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}
