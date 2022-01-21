%%writefile max2.cpp
#include <iostream>
#include <string>
#include "cublas_v2.h"
#include <cuda_runtime.h>
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

template<class T>
class Herk {
  private:
    int A_row, A_col, C_row, C_col;
    char mode;
    T *HostMatrixA;
    T *HostMatrixC;
    T *DeviceMatrixA;
    T *DeviceMatrixC;
    double  alpha;
    double beta;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    Herk(int A_row, int A_col, int C_row, int C_col, double alpha, double beta, char mode)
        : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), alpha(alpha), beta(beta), mode(mode) {}

    void FreeMemory(){
      if (HostMatrixA)
        delete[] HostMatrixA;
     
      if (HostMatrixC)
        delete[] HostMatrixC;
      
      cudaStatus = cudaFree(DeviceMatrixA);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for A" << std::endl;   
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

    template<class C>
    void PrintComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
      int row, col;
      for (row = 0; row < matrix_row; row++) {
        for (col = 0; col < matrix_col; col++) {
          std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
        }
        std::cout << "\n";
      } 
    }

    template<class C>
    void PrintComplexMatrixLow(C* Matrix, int matrix_row, int matrix_col) {
      int row, col;
      for (row = 0; row < matrix_row; row++) {
        for (col = 0; col < matrix_col; col++) {
          if (row >= col) {
            std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
          }        
        }
        std::cout << "\n";
      } 
    }

    template<class C>
    void InitializeComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
      int row, col;  
      for (col = 0; col < matrix_col; col++) {           
        for (row = 0; row < matrix_row; row++) {                      
          Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
          Matrix[INDEX(row, col, matrix_row)].y = 0.0f;              
        }
      }
    }

    template<class C>
    void InitializeComplexMatrixLow(C* Matrix, int matrix_row, int matrix_col) {
      int row, col;  
      for (col = 0; col < matrix_col; col++) {           
        for (row = 0; row < matrix_row; row++) {
          if (row >= col) {                      
            Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
            Matrix[INDEX(row, col, matrix_row)].y = 0.0f;
          }              
        }
      }
    }
   
  
    int HerkApiCall() {
      
      // Host Memory Allocation for Matrices
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
      
      // define an mxk matrix A, B, C column by column and based on mode passed
      // using RANDOM macro to generate random numbers between 0 - 100


      switch (mode) {
        case 'C': {
          InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          InitializeComplexMatrixLow<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          PrintComplexMatrixLow<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
          break; 
        }
                            
        case 'Z': {
          InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          InitializeComplexMatrixLow<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

          // printing input matrices
          //Lower triangle of Matrix C
          std::cout << "\nMatrix C:\n";
          PrintComplexMatrixLow<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);
          
          // printing matrix A column by column
          std::cout << "\nMatrix A:\n";
          PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
          break; 
        }
      }

      cudaStatus = cudaMalloc((void **)&DeviceMatrixA , A_row * A_col * sizeof(*HostMatrixA));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for A " << std::endl;
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
      
      status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);  // C -> d_C
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying matrix C from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
     
      switch (mode) {
        case 'C': {
          std::cout << "\nCalling Cherk API\n";
          clk_start = clock();

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                               A_row, A_col, (float *)&alpha, (cuComplex *)DeviceMatrixA, A_row, (float *)&beta, 
                               (cuComplex *)DeviceMatrixC, C_row);
        
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

          // matrix - matrix multiplication : d_C = alpha * d_A * d_B + beta * d_C
          // d_A - mxk matrix, d_B - kxn matrix, d_C - mxn matrix
          // alpha, beta - scalars
          status = cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                               A_row, A_col, (double *)&alpha, (cuDoubleComplex *)DeviceMatrixA, A_row, (double *)&beta, 
                               (cuDoubleComplex *)DeviceMatrixC, C_row);
        
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


      // Copying Matrices from device to host
      status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                              DeviceMatrixC, C_row, HostMatrixC, C_row);  // copy d_z -> C

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      std::cout << "\nMatriz C after " << mode << "Herk operation is:\n";

      switch (mode) {
        case 'C': {
          PrintComplexMatrixLow<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col); 
          break;
        }

        case 'Z': {
          PrintComplexMatrixLow<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col); 
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
  
  int A_row, A_col, C_row, C_col, status;
  double alpha, beta;
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

    else if (!(cmd_argument.compare("-alpha")))
      alpha = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta")))
      beta = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  //initializing values for C matrix
  C_row = A_row;
  C_col = A_row;
  
  // function call
  switch (mode) {
    case 'C': {
      Herk<cuComplex> Cherk(A_row, A_col, C_row, C_col, alpha, beta, mode);
      status = Cherk.HerkApiCall();
      break;
    }

    case 'Z': {
      Herk<cuDoubleComplex> Zherk(A_row, A_col, C_row, C_col, alpha, beta, mode);
      status = Zherk.HerkApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}





 

 
