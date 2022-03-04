%%writefile nrm2.cc
#include <unordered_map>
#include "nrm2.h"

template<class T>
Nrm2<T>::Nrm2(int vector_length, char mode)
              : vector_length(vector_length), mode(mode) {}

template<class T>
void Nrm2<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Nrm2<T>::Nrm2ApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];

  if (!HostVectorX) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorX)\n");
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
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host vector to Device vector using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
    
  /**
   * The Error values returned by API are :\n 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_ALLOC_FAILED - The reduction buffer could not be allocated \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to Nrm2 to compute the Euclidean norm of the vector x \n
   * The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow, with the result being 
   *   equivalent to \f$ ∑ni = 1 (x[j] × x[j]) \f$  √ where \f$ j = 1 + (i − 1) * incx \f$ in exact arithmetic \n  
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Snrm2 API\n";
      float result;
      clk_start = clock();

      status = cublasSnrm2(handle, vector_length, (float *)DeviceVectorX, 1, (float *)&result);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Snrm2 kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Snrm2 API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "nrm2 operation : " << abs((result));
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dnrm2 API\n";
      double result;
      clk_start = clock();

      status = cublasDnrm2(handle, vector_length, (double *)DeviceVectorX, 1, (double *)&result);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dnrm2 kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dnrm2 API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "nrm2 operation : " << abs((result));
      break;
    }

    case 'C': {
      std::cout << "\nCalling Scnrm2 API\n";
      float result;

      clk_start = clock();

      status = cublasScnrm2(handle, vector_length, (cuComplex *)DeviceVectorX, 1, (float *)&result);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Scnrm2 kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Scnrm2 API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "nrm2 operation : " << abs((result));
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Dznrm2 API\n";
      double result;
      clk_start = clock();

      status = cublasDznrm2(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1, (double *)&result);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dznrm2 kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dznrm2 API call ended\n";
      std::cout << "\nEuclidean norm of x after " << mode << "nrm2 operation : " << abs((result));
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

void mode_S(int vector_length) {          

  Nrm2<float> Snrm2(vector_length, 'S' );
  Snrm2.Nrm2ApiCall();
}

void mode_D(int vector_length) {
            
  Nrm2<double> Dnrm2(vector_length, 'D');
  Dnrm2.Nrm2ApiCall();
}

void mode_C(int vector_length) {

  Nrm2<cuComplex> Cnrm2(vector_length, 'C');
  Cnrm2.Nrm2ApiCall(); 
}

void mode_Z(int vector_length) {
            
  Nrm2<cuDoubleComplex> Znrm2(vector_length, 'Z');
  Znrm2.Nrm2ApiCall(); 
}


void (*cublas_func_ptr[])(int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int vector_length;
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
  

  (*cublas_func_ptr[mode_index[mode]])(vector_length);
  
  return 0;
}
