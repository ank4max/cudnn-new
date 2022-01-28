%%writefile ne.cpp
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublas_utility.h"

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end, operations)  ((1e-9 * 2 * operations) / (clk_end - clk_start))

template<class T>
class GemmBatched {
  private:
    int A_row, A_col, B_row, B_col, C_row, C_col, batch_count;
    char mode;
    T **HostMatrixA;
    T **HostMatrixB;
    T **HostMatrixC;
    T **HostPtrToDeviceMatA;
    T **HostPtrToDeviceMatB;
    T **HostPtrToDeviceMatC;
    T **DeviceMatrixA;
    T **DeviceMatrixB;
    T **DeviceMatrixC;
    T alpha;
    T beta;
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    GemmBatched(int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int batch_count, T alpha, T beta, char mode)
        : A_row(A_row), A_col(A_col), B_row(B_row), B_col(B_col),
          C_row(C_row), C_col(C_col), batch_count(batch_count), alpha(alpha), beta(beta), mode(mode) {}

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

    int GemmBatchedApiCall() {
      // Host Memory Allocation for Matrices
      HostMatrixA = new T*[batch_count];
      HostMatrixB = new T*[batch_count];
      HostMatrixC = new T*[batch_count];

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
      // using RANDOM macro to generate random numbers
      switch (mode) {
        case 'S': {
          util::InitializeStridedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
          util::InitializeStridedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedMatrix<float>((float **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedMatrix<float>((float **)HostMatrixB, B_row, B_col, batch_count);
          std::cout << "\nMatrix C:\n";
          util::PrintStridedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'D': {
          util::InitializeStridedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
          util::InitializeStridedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedMatrix<double>((double **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedMatrix<double>((double **)HostMatrixB, B_row, B_col, batch_count);
          std::cout << "\nMatrix C:\n";
          util::PrintStridedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);

          break;
        }

        case 'C': {
          util::InitializeStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
          util::InitializeStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixB, B_row, B_col, batch_count);
          std::cout << "\nMatrix C:\n";
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'Z': {
          util::InitializeStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
          util::InitializeStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixB, B_row, B_col, batch_count);
          std::cout << "\nMatrix C:\n";
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);

          break;
        }

        case 'H': {
          util::InitializeStridedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
          util::InitializeStridedMatrix<__half>((__half **)HostMatrixB, B_row, B_col, batch_count);
          util::InitializeStridedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);

          // printing input matrices
          std::cout << "\nMatrix A:\n";
          util::PrintStridedMatrix<__half>((__half **)HostMatrixA, A_row, A_col, batch_count);
          std::cout << "\nMatrix B:\n";
          util::PrintStridedMatrix<__half>((__half **)HostMatrixB, B_row, B_col, batch_count);
          std::cout << "\nMatrix C:\n";
          util::PrintStridedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);

          break;
        }
      }
      
      //allocating matrices on device    
      int batch;
      HostPtrToDeviceMatA = new T*[batch_count];
      HostPtrToDeviceMatB = new T*[batch_count];
      HostPtrToDeviceMatC = new T*[batch_count];

      
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

        cudaStatus = cudaMalloc((void**)&HostPtrToDeviceMatC[batch], C_row * C_col * sizeof(T));
        if (cudaStatus != cudaSuccess) {
          fprintf (stderr, "!!!! Device memory allocation for matrix (C) failed\n");
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

      cudaStatus = cudaMalloc((void**)&DeviceMatrixC, batch_count * sizeof(T*));
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Device memory allocation for matrix (C) failed\n");
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
      cudaStatus = cudaMemcpy(DeviceMatrixC, HostPtrToDeviceMatC, sizeof(T*) * batch_count, cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Memory copy on device for matrix (C) failed\n");
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
    
        status = cublasSetMatrix(C_row, C_col, sizeof(T), HostMatrixC[batch], C_row, HostPtrToDeviceMatC[batch], C_row);
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf (stderr, "!!!! Setting up values on device for Matrix C failed\n");
          FreeMemory();
          return EXIT_FAILURE;
        }
      }

      switch (mode) {
        case 'S': {
          std::cout << "\nCalling Sgemmbatched API\n";
          clk_start = clock();
 
          status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              A_row, B_col, A_col, (float *)&alpha, (float**)DeviceMatrixA,
                              A_row, (float**)DeviceMatrixB, B_row,
                              (float *)&beta, (float **)DeviceMatrixC, C_row, batch_count);


          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Sgemmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Sgemmbatched API call ended\n";
          break;
        }

        case 'D': {
          std::cout << "\nCalling Dgemmbatched API\n";
          clk_start = clock();

          status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              A_row, B_col, A_col, (double *)&alpha, (double**)DeviceMatrixA,
                              A_row, (double**)DeviceMatrixB, B_row,
                              (double *)&beta, (double **)DeviceMatrixC, C_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Dgemmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Dgemmbatched API call ended\n";
          break;
        }

        case 'H': {
          std::cout << "\nCalling Hgemmbatched API\n";
          clk_start = clock();

          status = cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              A_row, B_col, A_col, (__half *)&alpha, (__half **)DeviceMatrixA,
                              A_row, (__half **)DeviceMatrixB, B_row,
                              (__half *)&beta, (__half **)DeviceMatrixC, C_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Hgemmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Hgemmbatched API call ended\n";
          break;
        }

        case 'C': {
          std::cout << "\nCalling Cgemmbatched API\n";
          clk_start = clock();
       
          status = cublasCgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              A_row, B_col, A_col, (cuComplex *)&alpha, (cuComplex **)DeviceMatrixA,
                              A_row, (cuComplex **)DeviceMatrixB, B_row,
                              (cuComplex *)&beta, (cuComplex **)DeviceMatrixC, C_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Cgemmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Cgemmbatched API call ended\n";
          break;
        }

        case 'Z': {
          std::cout << "\nCalling Zgemmbatched API\n";
          clk_start = clock();
   
          status = cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              A_row, B_col, A_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex **)DeviceMatrixA,
                              A_row, (cuDoubleComplex **)DeviceMatrixB, B_row,
                              (cuDoubleComplex *)&beta, (cuDoubleComplex **)DeviceMatrixC, C_row, batch_count);

          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Zgemmbatched kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Zgemmbatched API call ended\n";
          break;
        }
      }

      // getting the final output
      for (batch = 0; batch < batch_count; batch++) {
        status = cublasGetMatrix(C_row, C_col, sizeof(T), HostPtrToDeviceMatC[batch], C_row, HostMatrixC[batch], C_row);
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf (stderr, "!!!! API execution failed\n");
          return EXIT_FAILURE;
        }
      }
  
      if (cudaStatus != cudaSuccess) {
        fprintf (stderr, "!!!! Failed to to Get values in Host Matrix C");
        return EXIT_FAILURE;
      }

      std::cout << "\nMatriz C after " << mode << "gemmbatched operation is:\n";

      switch (mode) {
        case 'S': {
          util::PrintStridedMatrix<float>((float **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'D': {
          util::PrintStridedMatrix<double>((double **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'C': {
          util::PrintStridedComplexMatrix<cuComplex>((cuComplex **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'Z': {
          util::PrintStridedComplexMatrix<cuDoubleComplex>((cuDoubleComplex **)HostMatrixC, C_row, C_col, batch_count);
          break;
        }

        case 'H': {
          util::PrintStridedMatrix<__half>((__half **)HostMatrixC, C_row, C_col, batch_count);
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

  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count, status;
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

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);
    
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

  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  

  // function call
  switch (mode) {
    case 'S': {
      float alpha = (float)alpha_real;
      float beta = (float)beta_real;
      GemmBatched<float> Sgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Sgemmbatched.GemmBatchedApiCall();
      break;
    }

    case 'D': {
      double alpha = alpha_real;
      double beta = beta_real;
      GemmBatched<double> Dgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Dgemmbatched.GemmBatchedApiCall();
      break;
    }

    case 'C': {
      cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
      cuComplex beta = {(float)beta_real, (float)beta_imaginary};
      GemmBatched<cuComplex> Cgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Cgemmbatched.GemmBatchedApiCall();
      break;
    }

    case 'Z': {
      cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
      cuDoubleComplex beta = {beta_real, beta_imaginary};
      GemmBatched<cuDoubleComplex> Zgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Zgemmbatched.GemmBatchedApiCall();
      break;
    }

    case 'H': {
      __half alpha = (__half)alpha_real;
      __half beta = (__half)beta_real;
      GemmBatched<__half> Hgemmbatched(A_row, A_col, B_row, B_col, C_row, C_col, batch_count, alpha, beta, mode);
      status = Hgemmbatched.GemmBatchedApiCall();
      break;
    }
  }

  return EXIT_SUCCESS;
}



