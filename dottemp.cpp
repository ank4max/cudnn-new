%%writefile nex.cpp
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
class Dot {
  private:
    int vector_length;
    char mode;
    T *HostVectorX;
    T *HostVectorY;
    T *DeviceVectorX;
    T *DeviceVectorY;
    T *dot_product;
    cudaError_t cudaStatus; 
    cublasStatus_t status; 
    cublasHandle_t handle;
    clock_t clk_start, clk_end;

  public:
    Dot(int vector_length, char mode)
        : vector_length(vector_length), mode(mode) {}

    void FreeMemory(){
      if (HostVectorX)
        delete[] HostVectorX;
      
      if (HostVectorY)
        delete[] HostVectorY;
      
      cudaStatus = cudaFree(DeviceVectorX);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for X" << std::endl;   
      }
      
      cudaStatus = cudaFree(DeviceVectorY);  // free device memory
      if (cudaStatus != cudaSuccess) {
        std::cout << " The device memory deallocation failed for Y" << std::endl;
      }
      
      status  = cublasDestroy(handle);  // destroy CUBLAS context
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Unable to uninitialize handle \n");
      }
    }

    int DotApiCall() {
      
      // Host Memory Allocation for Matrices
      HostVectorX = new T[vector_length]; 
      HostVectorY = new T[vector_length]; 
      
      if (!HostVectorX) {
        fprintf (stderr, "!!!! Host memory allocation error (Vector X)\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      if (!HostVectorY) {
        fprintf (stderr, "!!!! Host memory allocation error (Vector Y)\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
            
      // define an mxk matrix A, B, C column by column and based on mode passed
      // using RANDOM macro to generate random numbers between 0 - 100

      switch (mode) {
        case 'S': {
         //initializing vectors X and Y
         util::InitializeVector<float>((float *)HostVectorX, vector_length);
         util::InitializeVector<float>((float *)HostVectorY, vector_length); 
         
         //Print initial Vectors
         std::cout << "\nX vector initially:\n";
         util::PrintVector<float>((float *)HostVectorX, vector_length);
         std::cout << "\nY vector initially:\n";
         util::PrintVector<float>((float *)HostVectorY, vector_length);
         break;
        }

        case 'D': {
          //initializing vectors X and Y
          util::InitializeVector<double>((double *)HostVectorX, vector_length);
          util::InitializeVector<double>((double *)HostVectorY, vector_length); 
         
          //Print initial Vectors
          std::cout << "\nX vector initially:\n";
          util::PrintVector<double>((double *)HostVectorX, vector_length);
          std::cout << "\nY vector initially:\n";
          util::PrintVector<double>((double *)HostVectorY, vector_length);
          break;  
        }

        case 'C': {
          //initializing vectors X and Y
          util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
          util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length); 
         
          //Print initial Vectors
          std::cout << "\nX vector initially:\n";
          util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
          std::cout << "\nY vector initially:\n";
          util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
          
          break; 
        }
                            
        case 'Z': {
          //initializing vectors X and Y
          util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
          util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length); 
         
          //Print initial Vectors
          std::cout << "\nX vector initially:\n";
          util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
          std::cout << "\nY vector initially:\n";
          util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
          break; 
        }
      }

      cudaStatus = cudaMalloc((void **)&DeviceVectorX, vector_length * sizeof(*HostVectorX));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for X " << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      cudaStatus = cudaMalloc((void **)&DeviceVectorY, vector_length * sizeof(*HostVectorY));
      if(cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for Y " << std::endl;
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
      status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);   // A -> d_A
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying vector x from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);  // B -> d_B
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "Copying vector y from host to device failed\n");
        FreeMemory();
        return EXIT_FAILURE;
      }
      
      switch (mode) {
        case 'S': {
          std::cout << "\nCalling Sdot API\n";
          clk_start = clock();

          // performing dot product operation and storing result in dot_product variable
          status = cublasSdot(handle, vector_length, (float *)DeviceVectorX, 1, (float *)DeviceVectorY, 1, (float *)&dot_product);

        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Sdot kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Sdot API call ended\n";
          break;
        }
                            
        case 'D': {
          std::cout << "\nCalling Ddot API\n";
          clk_start = clock();

          // performing dot product operation and storing result in dot_product variable
          status = cublasDdot(handle, vector_length, (double *)DeviceVectorX, 1, (double *)DeviceVectorY, 1, (double *)&dot_product);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Ddot kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Ddot API call ended\n";
          break;
        }

        case 'C': {
          std::cout << "\nCalling Cdot\n";
          clk_start = clock();

          // performing dot product operation and storing result in dot_product variable
          status = cublasCdotu(handle, vector_length, (cuComplex *)DeviceVectorX, 1, (cuComplex *)DeviceVectorY, 1, (cuComplex *)&dot_product);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Cdotu kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Cdot API call ended\n";
          break;
        }
      
        case 'Z': {
          std::cout << "\nCalling Zdot API\n";
          clk_start = clock();

          // performing dot product operation and storing result in dot_product variable
          status = cublasZdotu(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1, (cuDoubleComplex *)DeviceVectorY, 1, (cuDoubleComplex *)&dot_product);
        
          if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "!!!!  Zdot kernel execution error\n");
            FreeMemory();
            return EXIT_FAILURE;
          }

          clk_end = clock();
          std::cout << "Zdot API call ended\n";
          break;
        }
      }
      
      std::cout << "\nThe dot product after " << mode << "dot operation is :";
      switch (mode) {
        case 'S': {
          std::cout << "\nDot product X.Y is : " << dot_product << "\n"; 
          break;
        }

        case 'D': {
          std::cout << "\nDot product X.Y is : " << dot_product << "\n";   
          break;
        }

        case 'C': {
          std::cout << "\nDot product X.Y is : " << dot_product.x << "+" << dot_product.y << "*I "<<"\n";  
          break;
        }

        case 'Z': {
          std::cout << "\nDot product X.Y is : " << dot_product.x << "+" << dot_product.y << "*I "<<"\n";   
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
  
  int vector_length, status;
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
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  // function call
  switch (mode) {
    case 'S': {
      
      Dot<float> Sdot(vector_length, mode);
      status = Sdot.DotApiCall();
      break;
    }

    case 'D': {
     
      Dot<double> Ddot(vector_length, mode);
      status = Ddot.DotApiCall();
      break;
    }

    case 'C': {
 
      Dot<cuComplex> Cdot(vector_length, mode);
      status = Cdot.DotApiCall();
      break;
    }

    case 'Z': {
      
      Dot<cuDoubleComplex> Zdot(vector_length, mode);
      status = Zdot.DotApiCall();
      break;
    }          
  }

  return EXIT_SUCCESS;
}





 

 
