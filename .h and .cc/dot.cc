#include <unordered_map>
#include "cublas_dot_test.h"


//! Dot constructor - to initialize the global varibles using initializer list
template<class T>
Dot<T>::Dot(int vector_length, char mode)
    : vector_length(vector_length), mode(mode) {}

//! FreeMemory function - to free the allocated memory when program is ended or in case of any error
template<class T>
void Dot<T>::FreeMemory() {
  //! Free Host Memory
  if (HostVectorX)
    delete[] HostVectorX;
  
  if (HostVectorY)
    delete[] HostVectorY;
  
  //! Free Device Memory
  cudaStatus = cudaFree(DeviceVectorX);  // free device memory
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;   
  }
  
  cudaStatus = cudaFree(DeviceVectorY);  // free device memory
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Y" << std::endl;
  }
  
  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);  // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}


template<class T>
int Dot<T>::DotApiCall() {
  //! Allocating Host Memory for Matrices
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
        
  /**
   * Switch Case - To Initialize and Print input vectors based on mode passed
   * X and Y are general vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;  
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);          
      break; 
    }

    case 'H': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);      
      break; 
    }
                        
    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
      break; 
    }

    case 'T': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
      break; 
    }
  }
  
  //! Allocating Device Memory for Vectors using cudaMalloc()
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

  //! Initializing CUBLAS context
  status = cublasCreate(&handle);      
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host vector to Device vector using cublasSetMatrix()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);   
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector x from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector y from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to performs Dot product between 2 vectors : 
   * dot_product = X.Y
   * The result is ∑ni=1(x[k]×y[j]), where k=1+(i−1)* incx and j=1+(i−1)* incy 
   * Notice that in the first equation the conjugate of the element of vector x should be used if the function name ends in 
        character ‘c’ and that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below :
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized
   * CUBLAS_STATUS_ALLOC_FAILED - The reduction buffer could not be allocated
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU
   */ 
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sdot API\n";
      clk_start = clock();
      float dot_product;

      status = cublasSdot(handle, vector_length, (float *)DeviceVectorX, 1, 
                          (float *)DeviceVectorY, 1, (float *)&dot_product);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sdot kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sdot API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product << "\n"; 
      break;
    }
                        
    case 'D': {
      std::cout << "\nCalling Ddot API\n";
      double dot_product;
      clk_start = clock();

      status = cublasDdot(handle, vector_length, (double *)DeviceVectorX, 1, 
                          (double *)DeviceVectorY, 1, (double *)&dot_product);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ddot kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ddot API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product << "\n"; 
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cdotu\n";
      cuComplex dot_product;
      clk_start = clock();

      status = cublasCdotu(handle, vector_length, (cuComplex *)DeviceVectorX, 1, 
                            (cuComplex *)DeviceVectorY, 1, (cuComplex *)&dot_product);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cdotu kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cdotu API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product.x << "+" 
                                             << dot_product.y << "*I "<<"\n";  
      break;
    }

    case 'H': {
      std::cout << "\nCalling Cdotc\n";
      cuComplex dot_product;
      clk_start = clock();

      status = cublasCdotc(handle, vector_length, (cuComplex *)DeviceVectorX, 1, 
                            (cuComplex *)DeviceVectorY, 1, (cuComplex *)&dot_product);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cdotc kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cdotc API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product.x << "+" 
                                             << dot_product.y << "*I "<<"\n";  
      break;
    }
  
    case 'Z': {
      std::cout << "\nCalling Zdotu API\n";
      cuDoubleComplex dot_product;
      clk_start = clock();

      status = cublasZdotu(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1,
                            (cuDoubleComplex *)DeviceVectorY, 1, (cuDoubleComplex *)&dot_product);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zdotu kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zdotu API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product.x << "+" 
                                             << dot_product.y << "*I "<<"\n";  
      break;
    }

    case 'T': {
      std::cout << "\nCalling Zdotc API\n";
      cuDoubleComplex dot_product;
      clk_start = clock();

      status = cublasZdotc(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1,
                            (cuDoubleComplex *)DeviceVectorY, 1, (cuDoubleComplex *)&dot_product);
    
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zdotc kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zdotc API call ended\n";
      std::cout << "\nDot product X.Y is : " << dot_product.x << "+" 
                                             << dot_product.y << "*I "<<"\n";  
      break;
    }
  }
                      
  //! Printing latency and throughput of the API
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, vector_length) << "\n\n";
  
  FreeMemory();

  return EXIT_SUCCESS; 
}

void mode_S(int vector_length) {

   Dot<float> Sdot(vector_length, 'S');
   Sdot.DotApiCall();
}

void mode_D(int vector_length) {
  
  Dot<double> Ddot(vector_length, 'D');
  Ddot.DotApiCall();
}

void mode_C(int vector_length) {
  
  Dot<cuComplex> Cdotu(vector_length, 'C');
  Cdotu.DotApiCall();
}

void mode_H(int vector_length) {
  
  Dot<cuComplex> Cdotc(vector_length, 'H');
  Cdotc.DotApiCall();
}

void mode_Z(int vector_length) {
  
  Dot<cuDoubleComplex> Zdotu(vector_length, 'Z');
  Zdotu.DotApiCall();
}

void mode_T(int vector_length) {
  
  Dot<cuDoubleComplex> Zdotc(vector_length, 'T');
  Zdotc.DotApiCall();
}

void (*cublas_func_ptr[])(int) = {mode_S, mode_D, mode_C, mode_H, mode_Z, mode_T};
        
int main(int argc, char **argv) {
  int vector_length, status;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['H'] = 3;
  mode_index['Z'] = 4;
  mode_index['T'] = 5;

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
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  (*cublas_func_ptr[mode_index[mode]])(vector_length);

  return EXIT_SUCCESS;
}
