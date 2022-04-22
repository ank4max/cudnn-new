%%writefile n.cc
#include <unordered_map>
#include "cublas_Nrm2Ex_test.h"

template<class T>
Nrm2Ex<T>::Nrm2Ex(int vector_length, char mode)
    : vector_length(vector_length), mode(mode) {}

template<class T>
void Nrm2Ex<T>::FreeMemory() {
  //! Free Host Memory
  if (HostVectorX)
    delete[] HostVectorX;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceVectorX);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Nrm2Ex<T>::Nrm2ExApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];

  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vector based on mode passed,
   * X is a vector 
   */
  switch (mode) {
    case 'S': {
      util::InitializeVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);        
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);      
      break;
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      break;
    }

    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);       
      break;
    }

    case 'H': {
      util::InitializeVector<__half>((__half *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<__half>((__half *)HostVectorX, vector_length);        
      break;
    }
  }
  
  //! Allocating Device Memory for Vector using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceVectorX, vector_length * sizeof(*HostVectorX));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X " << std::endl;
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
  
  //! Copying values of Host vector to Device vector using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * API call to Nrm2Ex to compute the Euclidean norm of the vector x \n.
   * This function is an API generalization of the routine cublas<t>nrm2 where input data, output data and compute type can be specified  
     independently.
   * The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow, with the result being 
   * equivalent to \f$ ∑ni = 1 (x[j] × x[j]) \f$  √ where \f$ j = 1 + (i − 1) * incx \f$ in exact arithmetic \n  
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */
    
  /**
   * The Error values returned by API are :\n 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_ALLOC_FAILED - The reduction buffer could not be allocated \n
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters xType, resultType and executionType is not supported \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SNrm2Ex API\n";
      float result;
      clk_start = clock();

      status = cublasNrm2Ex(handle, vector_length, (float *)DeviceVectorX, CUDA_R_32F, VECTOR_LEADING_DIMENSION, (float *)&result,
                            CUDA_R_32F, CUDA_R_32F);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SNrm2Ex kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SNrm2Ex API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "Nrm2Ex operation : " << abs(result);
      break;
    }

    case 'D': {
      std::cout << "\nCalling DNrm2Ex API\n";
      double result;
      clk_start = clock();

      status = cublasNrm2Ex(handle, vector_length, (double *)DeviceVectorX, CUDA_R_64F, VECTOR_LEADING_DIMENSION, (double *)&result,
                             CUDA_R_64F, CUDA_R_64F);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DNrm2Ex kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DNrm2Ex API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "Nrm2Ex operation : " << abs(result);
      break;
    }

    case 'C': {
      std::cout << "\nCalling CNrm2Ex API\n";
      float result;

      clk_start = clock();

      status = cublasNrm2Ex(handle, vector_length, (cuComplex *)DeviceVectorX, CUDA_C_32F, VECTOR_LEADING_DIMENSION, (float *)&result
                            ,CUDA_C_32F, CUDA_C_32F);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CNrm2Ex kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ScNrm2Ex API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "Nrm2Ex operation : " << abs(result);
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZNrm2Ex API\n";
      double result;
      clk_start = clock();

      status = cublasNrm2Ex(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, CUDA_C_64F, VECTOR_LEADING_DIMENSION, (double *)&result, CUDA_C_64F, CUDA_C_64F);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZNrm2Ex kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZNrm2Ex API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "Nrm2Ex operation : " << abs(result);
      break;
    }

    case 'H': {
      std::cout << "\nCalling HNrm2Ex API\n";
      __half result;
      clk_start = clock();

      status = cublasNrm2Ex(handle, vector_length, (__half *)DeviceVectorX, CUDA_R_16F, VECTOR_LEADING_DIMENSION, (__half *)&result,
                            CUDA_R_16F, CUDA_R_32F);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!! HNrm2Ex kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "HNrm2Ex API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "Nrm2Ex operation : " << abs(result);
      break;
    }
  }

  long long total_operations = vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int vector_length) {
  Nrm2Ex<float> SNrm2Ex(vector_length, 'S' );
  return SNrm2Ex.Nrm2ExApiCall();
}

int mode_D(int vector_length) {
  Nrm2Ex<double> DNrm2Ex(vector_length, 'D');
  return DNrm2Ex.Nrm2ExApiCall();
}

int mode_C(int vector_length) {
  Nrm2Ex<cuComplex> CNrm2Ex(vector_length, 'C');
  return CNrm2Ex.Nrm2ExApiCall(); 
}

int mode_Z(int vector_length) {
  Nrm2Ex<cuDoubleComplex> ZNrm2Ex(vector_length, 'Z');
  return ZNrm2Ex.Nrm2ExApiCall(); 
}

int mode_H(int vector_length) {
  Nrm2Ex<__half> HNrm2Ex(vector_length, 'H' );
  return HNrm2Ex.Nrm2ExApiCall();
}

int (*cublas_func_ptr[])(int) = {
  mode_S, mode_D, mode_C, mode_Z, mode_H
};

int main(int argc, char **argv) {
  int vector_length, status;
  char mode;
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['Z'] = 3;
  mode_index['H'] = 4;

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

  //! Dimension check
  if(vector_length <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  status = (*cublas_func_ptr[mode_index[mode]])(vector_length);
  
  return status;
}
