%%writefile m1.cc
#include <unordered_map>
#include "cublas_DotEx_test.h"


//! DotEx constructor - to initialize the global varibles using initializer list
template<class T>
DotEx<T>::DotEx(int vector_length, char mode)
    : vector_length(vector_length), mode(mode) {}

//! FreeMemory function - to free the allocated memory when program is ended or in case of any error
template<class T>
void DotEx<T>::FreeMemory() {
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
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}


template<class T>
int DotEx<T>::DotExApiCall() {
  //! Allocating Host Memory for Matrices
  HostVectorX = new T[vector_length]; 
  HostVectorY = new T[vector_length]; 
  
  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (Vector X)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorY) {
    std::cout << "!!!! Host memory allocation error (Vector Y)\n";
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

    case 'C': {
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length); 
      
      std::cout << "\nX vector initially:\n";
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nY vector initially:\n";
      util::PrintVector<float>((float *)HostVectorY, vector_length);
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
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host vector to Device vector using cublasSetMatrix()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);   
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector x from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector y from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to performs DotEx product between 2 vectors : \n
   * DotEx_product = X.Y \n
   * These functions are an API generalization of the routines cublas<t>DotEx and cublas<t>DotExc where input data, output data and compute  
     type can be specified independently. Note: cublas<t>DotExc is DotEx product conjugated, cublas<t>DotExu is DotEx product unconjugated.
   * The result is \f$  ∑_{i = 1}^n(x[k] × y[j]) \f$, where \f$ k = 1 + (i − 1) * incx \f$ and \f$ j = 1 + (i − 1) * incy \f$\n
   * Notice that in the first equation the conjugate of the element of vector x should be used if the function name ends in \n
        character ‘c’ and that the last two equations reflect 1-based indexing used for compatibility with Fortran. \n
   */
  
  /**
   * The possible error values returned by this API and their meanings are listed below : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized  \n
   * CUBLAS_STATUS_ALLOC_FAILED - The reduction buffer could not be allocated \n
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters xType,yType, resultType and executionType is not supported. \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU  \n
   */ 
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling DotEx API\n";
      clk_start = clock();
      float DotEx_product;

      status = cublasDotEx(handle, vector_length, (float *)DeviceVectorX, CUDA_R_32F, VECTOR_LEADING_DIMENSION, 
                          (float *)DeviceVectorY, CUDA_R_32F, VECTOR_LEADING_DIMENSION, (float *)&DotEx_product, CUDA_R_32F,
                          CUDA_R_32F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DotEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DotEx API call ended\n";
      std::cout << "\nDotEx product X.Y is : " << DotEx_product << "\n"; 
      break;
    }

    case 'C': {
      std::cout << "\nCalling DotcEx API\n";
      clk_start = clock();
      float DotEx_product;

      status = cublasDotcEx(handle, vector_length, (float *)DeviceVectorX, CUDA_R_32F, VECTOR_LEADING_DIMENSION, 
                          (float *)DeviceVectorY, CUDA_R_32F, VECTOR_LEADING_DIMENSION, (float *)&DotEx_product, CUDA_R_32F,
                          CUDA_R_32F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DotcEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DotcEx API call ended\n";
      std::cout << "\nDotEx product X.Y is : " << DotEx_product << "\n"; 
      break;
    }                   
    
  }
                      
  //! Printing latency and throughput of the API
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / (double)(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, vector_length) << "\n\n";
  
  FreeMemory();

  return EXIT_SUCCESS; 
}

int mode_S(int vector_length) {

  DotEx<float> SDotEx(vector_length, 'S');
  return SDotEx.DotExApiCall();
}

int mode_C(int vector_length) {

  DotEx<float> CDotEx(vector_length, 'C');
  return CDotEx.DotExApiCall();
}



int (*cublas_func_ptr[])(int) = {mode_S, mode_C};
        
int main(int argc, char **argv) {
  int vector_length, status;
  char mode;
  
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['C'] = 1;
  

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
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
      
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! Dimension check
  if (vector_length <= 0){
    std::cout << "Minimum Dimension error\n";
    return EXIT_FAILURE;
  }

  status = (*cublas_func_ptr[mode_index[mode]])(vector_length);

  return status;
}
