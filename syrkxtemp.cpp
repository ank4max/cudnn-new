%%writefile nay1.cpp
#include <iostream>
#include <string>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "cublas_utility.h"
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

template<class T>
class Syrkx {
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
    Syrkx(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, T alpha, T beta, char mode)
        : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col), C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

    void FreeMemory(){
      if (HostMatrixA)
        delete[] HostMatrixA;
      
      if (HostMatrixB)
        delete[] HostMatrixB;

      if (HostMatrixC)
        delete[] HostMatrixC;
      
      cudaStatus = cudaFree(DeviceMatrixA);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;   
      }
      
      cudaStatus = cudaFree(DeviceMatrixB);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for B" << std::endl;
      }
      
      cudaStatus = cudaFree(DeviceMatrixC);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for C" << std::endl;
      }

      status  = cublasDestroy(handle);  // destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }

    int SyrkxApiCall() {
      
      // Host Memory Allocation for Matrices
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
      
      // define an mxk matrix A, B, C column by column and based on mode passed
      // using RANDOM macro to generate random numbers between 0 - 100


      switch (mode) {
        case 'S': {
          util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
          util::InitializeMatrix<float>((float *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
          
          // printing matrix B column by column
          std::cout << "\nMatrix B:\n";
          util::PrintMatrix<float>((float *)HostMatrixB, B_row, B_col);
          
          break;
        }

        case 'D': {
          util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
          util::InitializeMatrix<double>((double *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col); 
          
          // printing matrix B column by column
          std::cout << "\nMatrix B:\n";
          util::PrintMatrix<double>((double *)HostMatrixB, B_row, B_col); 
          break;  
        }

        case 'C': {
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          
          // printing matrix B column by column
          std::cout << "\nMatrix B:\n";
          util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixB, B_row, B_col);
          break; 
        }
                            
        case 'Z': {
          util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
          util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          
          // printing matrix B column by column
          std::cout << "\nMatrix B:\n";
          util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixB, B_row, B_col);
          break; 
        }
      }

      cudaStatus = cudaMalloc((void **)&DeviceMatrixA , A_row * A_col * sizeof(*HostMatrixA));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for A " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      cudaStatus = cudaMalloc((void **)&DeviceMatrixB , B_row * B_col * sizeof(*HostMatrixB));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for B " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      cudaStatus = cudaMalloc((void **)&DeviceMatrixC , C_row * C_col * sizeof(*HostMatrixC));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for C " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;   
      }

      // initialize CUBLAS context
      status = cublasCreate(&handle);      
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to initialize handle\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      // copy matrices from the host to the device
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
      

      switch (mode) {
        case 'S': {
          std::cout << "\nCalling Ssyrkx API\n";
          clk_start = clock();

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasSsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (float *)&alpha, (float *)DeviceMatrixA, A_row, (float *)DeviceMatrixB, 
                                B_row, (float *)&beta, (float *)DeviceMatrixC, C_row);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Ssyrkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Ssyrkx API call ended\n";
          break;
        }
                            
        case 'D': {
          std::cout << "\nCalling Dsyrkx API\n";
          clk_start = clock();

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasDsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (double *)&alpha, (double *)DeviceMatrixA, A_row, (double *)DeviceMatrixB, 
                                B_row, (double *)&beta, (double *)DeviceMatrixC, C_row);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Dsyrkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Dsyrkx API call ended\n";
          break;
        }

        case 'C': {
          std::cout << "\nCalling Csyrkx API\n";
          clk_start = clock();

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasCsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row, (cuComplex *)DeviceMatrixB, 
                                B_row, (cuComplex *)&beta, (cuComplex *)DeviceMatrixC, C_row);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Csyrkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Csyrkx API call ended\n";
          break;
        }
      
        case 'Z': {
          std::cout << "\nCalling Zsyrkx API\n";
          clk_start = clock();

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasZsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                A_row, A_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA, A_row, (cuDoubleComplex *)DeviceMatrixB, 
                                B_row, (cuDoubleComplex *)&beta, (cuDoubleComplex *)DeviceMatrixC, C_row);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Zsyrkx kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Zsyrkx API call ended\n";
          break;
        }
      }


      // Copying Matrices from device to host
      status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                              DeviceMatrixC, C_row, HostMatrixC, C_row);  // copy d_z -> C

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      std::cout << "\nMatriz C after " << mode << "Syrkx operation is:\n";

      switch (mode) {
        case 'S': {
          util::PrintSymmetricMatrix<float>((float *)HostMatrixC, C_row, C_col); 
          break;
        }

        case 'D': {
          util::PrintSymmetricMatrix<double>((double *)HostMatrixC, C_row, C_col);  
          break;
        }

        case 'C': {
          util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
          break;
        }

        case 'Z': {
          util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
          break;
        }
      }

      // printing latency and throughput of the function
      std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
                  "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
      
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

  // reading cmd line arguments
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
    
    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  B_row = A_row;
  B_col = A_col;
  C_row = A_row;
  C_col = A_row;
  
  // function call
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;
      float beta = (float)beta_real;

      Syrkx<float> Ssyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Ssyrkx.SyrkxApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;
      double beta = beta_real;

      Syrkx<double> Dsyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Dsyrkx.SyrkxApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
      cuComplex beta = {(float)beta_real, (float)beta_imaginary};

      Syrkx<cuComplex> Csyrkx(A_row, A_col, B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Csyrkx.SyrkxApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
      cuDoubleComplex beta = {beta_real, beta_imaginary};

      Syrkx<cuDoubleComplex> Zsyrkx(A_row, A_col,B_row, B_col, C_row, C_col, alpha, beta, mode);
      status = Zsyrkx.SyrkxApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}





 

 
