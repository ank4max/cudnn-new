#include <unordered_map>
#include "cublas_amin_test.h"

template<class T>
Amin<T>::Amin(int vector_length, char mode)
              : vector_length(vector_length), mode(mode) {}

template<class T>
void Amin<T>::FreeMemory() {
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
int Amin<T>::AminApiCall() {
  //! Allocating Host Memory for Vector
  HostVectorX = new T[vector_length];

  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vector based on mode passed,
   * X is a general vector 
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
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  

  /**
   * The Error values returned by API are : 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully 
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized 
   * CUBLAS_STATUS_ALLOC_FAILED - the reduction buffer could not be allocated
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU 
   */
  /**
   * API call to finds the (smallest) index of the element of the minimum magnitude
   */
  int result;
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling IsAmin API\n";
      clk_start = clock();

      status = cublasIsamin(handle, vector_length, (float *)DeviceVectorX,
                            VECTOR_LEADING_DIMENSION, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  IsAmin kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "IsAmin API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling IdAmin API\n";
      clk_start = clock();

      status = cublasIdamin(handle, vector_length, (double *)DeviceVectorX,
                            VECTOR_LEADING_DIMENSION, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  IdAmin kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "IdAmin API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling IcAmin API\n";
      clk_start = clock();

      status = cublasIcamin(handle, vector_length, (cuComplex *)DeviceVectorX, 
                            VECTOR_LEADING_DIMENSION, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  IcAmin kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "IcAmin API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling IzAmin API\n";
      clk_start = clock();

      status = cublasIzamin(handle, vector_length, (cuDoubleComplex *)DeviceVectorX,
                            VECTOR_LEADING_DIMENSION, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  IzAmin kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "IzAmin API call ended\n";
      break;
    }
  }
  
  //! Printing the result stored
  std::cout << "\nThe result obtained  after I" << mode << "Amin operation is:\n";

  std::cout << "\nMinimum value is at index - " << result << std::endl;

  long long total_operations =  vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int vector_length) { 
  Amin<float> SAmin(vector_length, 'S' );
  return SAmin.AminApiCall();
}

int mode_D(int vector_length) {
  Amin<double> DAmin(vector_length, 'D');
  return DAmin.AminApiCall();
}

int mode_C(int vector_length) {
  Amin<cuComplex> CAmin(vector_length, 'C');
  return CAmin.AminApiCall(); 
}

int mode_Z(int vector_length) {
  Amin<cuDoubleComplex> ZAmin(vector_length, 'Z');
  return ZAmin.AminApiCall(); 
}

int (*cublas_func_ptr[])(int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int vector_length, status;
  char mode;
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['Z'] = 3;

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
  if (vector_length <= 0){
      std::cout << "Minimum Dimension error\n";
      return EXIT_FAILURE;
  }
 
  status = (*cublas_func_ptr[mode_index[mode]])(vector_length);
  
  return status;
}
