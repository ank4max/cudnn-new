#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end, operations)  ((1e-9 * 2 * operations) / (clk_end - clk_start))

template<class T>
class TrsmBatched {
  private:
    int A_row, A_col, B_row, B_col, batch_count;
    char mode;
    T **HostMatrixA;
    T **HostMatrixB;
    T **HostPtrToDeviceMatA;
    T **HostPtrToDeviceMatB;
    T **DeviceMatrixA;
    T **DeviceMatrixB;
    T alpha;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    TrsmBatched(int A_row, int A_col, int B_row, int B_col, int batch_count, T alpha, char mode)
        : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
          batch_count(batch_count), alpha(alpha), mode(mode) {}

    void FreeMemory(){
      if (HostMatrixA)
        delete[] HostMatrixA;

      if (HostMatrixB)
        delete[] HostMatrixB;

      cudaStatus = cudaFree(DeviceMatrixA);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;
      }

      cudaStatus = cudaFree(DeviceMatrixB);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for B" << std::endl;
      }

      status  = cublasDestroy(handle);  // destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }

    int TrsmBatchedApiCall() {
      // Host Memory Allocation for Matrices
      HostMatrixA = new T*[batch_count];
      HostMatrixB = new T*[batch_count];

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

      // setting up values for matrices A and B
      // A is a triangular Matrix 
      // using RANDOM macro to generate random numbers
      switch (mode) {
        case 'S': {
          util::InitializeStridedTriangularMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }

        case 'D': {
          util::InitializeStridedTriangularMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }

        case 'C': {
          util::InitializeStridedTriangularComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }

        case 'Z': {
          util::InitializeStridedTriangularComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }
      }
      
      // allocating matrices on device    
      int batch;
      HostPtrToDeviceMatA = new T*[batch_count];
      HostPtrToDeviceMatB = new T*[batch_count];
      
      for(batch = 0; batch < batch_count; batch++) {
        cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatA[batch], A_row * A_col * sizeof(T));
        if (cudaStatus != cudaSuccess) {
          fprintf (stderr, "!!!! Device memory allocation for matrix (A) failed\n");
          FreeMemory();
          return EXIT_FAILURE;
        }

        cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatB[batch], B_row * B_col * sizeof(T));
        if (cudaStatus != cudaSuccess) {
          fprintf (stderr, "!!!! Device memory allocation for matrix (B) failed\n");
          FreeMemory();
          return EXIT_FAILURE;
        }
      }

      cudaStatus = cudaMalloc((void**)&DeviceMatrixA, batch_count * sizeof(T*));
        if (cudaStatus != cudaSuccess) {
          fprintf (stderr, "!!!! Device memory allocation for matrix (A) failed\n");
          FreeMemory();
          return EXIT_FAILURE;
      }

      cudaStatus = cudaMalloc((void**)&DeviceMatrixB, batch_count * sizeof(T*));
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Device memory allocation for matrix (B) failed\n");
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

      // setting the values of matrices on device
      cudaStatus = cudaMemcpy(DeviceMatrixA, HostPtrToDeviceMatA, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Memory copy on device for matrix (A) failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      cudaStatus = cudaMemcpy(DeviceMatrixB, HostPtrToDeviceMatB, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Memory copy on device for matrix (B) failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
     
      for (batch = 0; batch < batch_count; batch++) {
        status = cublasSetMatrix(A_row, A_col, sizeof(T), HostMatrixA[batch], A_row, HostPtrToDeviceMatA[batch], A_row);
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf (stderr, "!!!! Setting up values on device for Matrix A failed\n");
          FreeMemory();
          return EXIT_FAILURE;
        }

        status = cublasSetMatrix(B_row, B_col, sizeof(T), HostMatrixB[batch], B_row, HostPtrToDeviceMatB[batch], B_row);
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf (stderr, "!!!! Setting up values on device for Matrix B failed\n");
          FreeMemory();
          return EXIT_FAILURE;
        }
      }

      // solve d_A[batch] * X[batch] = alpha * d_B[batch]
      // d_A[batch] - m x m triangular matrix in lower mode
      // d_B[batch], X[batch] - m x n general matrices
      // alpha - scalar
      switch (mode) {
        case 'S': {
          std::cout << "\nCalling Strsmbatched API\n";
          clk_start = clock();
          
          status = cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col,
                                      (const float *)&alpha, (float* const*)DeviceMatrixA, A_row, 
                                      (float* const*)DeviceMatrixB, B_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Strsmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Strsmbatched API call ended\n";
          break;
        }

        case 'D': {
          std::cout << "\nCalling Dtrsmbatched API\n";
          clk_start = clock();

          status = cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                      (const double *)&alpha, (double* const*)DeviceMatrixA, A_row, 
                                      (double* const*)DeviceMatrixB, B_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Dtrsmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Dtrsmbatched API call ended\n";
          break;
        }
          
        case 'C': {
          std::cout << "\nCalling Ctrsmbatched API\n";
          clk_start = clock();
       
          status = cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                      (const cuComplex *)&alpha, 
                                      (cuComplex* const*)DeviceMatrixA, A_row, 
                                      (cuComplex* const*)DeviceMatrixB, B_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Ctrsmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Ctrsmbatched API call ended\n";
          break;
        }

        case 'Z': {
          std::cout << "\nCalling Ztrsmbatched API\n";
          clk_start = clock();
   
          status = cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, B_col, 
                                      (const cuDoubleComplex *)&alpha, 
                                      (cuDoubleComplex* const*)DeviceMatrixA, A_row, 
                                      (cuDoubleComplex* const*)DeviceMatrixB, B_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Ztrsmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Ztrsmbatched API call ended\n";
          break;
        }
      }

      // getting the final output
      for (batch = 0; batch < batch_count; batch++) {
        status = cublasGetMatrix(B_row, B_col, sizeof(T), HostPtrToDeviceMatB[batch],
                                 B_row, HostMatrixB[batch], B_row);                         
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf (stderr, "!!!! API execution failed\n");
          return EXIT_FAILURE;
        }
      }
  
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Failed to to Get values in Host Matrix B");
        return EXIT_FAILURE;
      }

      std::cout << "\nMatriz B after " << mode << "trsmbatched operation is:\n";

      switch (mode) {
        case 'S': {
          util::PrintStridedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }

        case 'D': {
          util::PrintStridedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
          break;
        }

        case 'C': {
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row,
                                                     B_col, batch_count);
          break;
        }

        case 'Z': {
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB,
                                                           B_row, B_col, batch_count);
          break;
        }
      }

      long long total_operations = A_row * A_col * B_col;

      // printing latency and throughput of the function
      std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
                   "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

      FreeMemory();

      return EXIT_SUCCESS;
    }
};

int main(int argc, char **argv) {
  int A_row, A_col, B_row, B_col, batch_count, status;
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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-batch_count"))) 
      batch_count = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

   A_col = A_row;
   B_row = A_col;

  // function call
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;
      TrsmBatched<float> Strsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, mode);
      status = Strsmbatched.TrsmBatchedApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;
      TrsmBatched<double> Dtrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, mode);
      status = Dtrsmbatched.TrsmBatchedApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
      TrsmBatched<cuComplex> Ctrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, mode);
      status = Ctrsmbatched.TrsmBatchedApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
      TrsmBatched<cuDoubleComplex> Ztrsmbatched(A_row, A_col, B_row, B_col, batch_count, alpha, mode);
      status = Ztrsmbatched.TrsmBatchedApiCall();
      break;
    }
  }

  return EXIT_SUCCESS;
}