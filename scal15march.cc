%%writefile max44.cc
#include <unordered_map>
#include "scal.h"

template<class T, class C>
Scal<T, C>::Scal(int vector_length, C alpha, char mode)
              : vector_length(vector_length), alpha(alpha), mode(mode) {}

template<class T, class C>
void Scal<T, C>::FreeMemory() {
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
    std::cout << " Unable to uninitialize handle \n";
  }
}

template<class T, class C>
int Scal<T, C>::ScalApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];

  if (!HostVectorX) {
    std::cout << " Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vectors based on mode passed,
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

    case 'H': {
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

    case 'T': {
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
    std::cout << " Failed to initialize handle\n";
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
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to scale the vector x by the scalar α and overwrites it with the resul: \f$ X = alpha * X \f$ \n
   * The performed operation is \f$ x[j] = α × x[j] for i = 1, …, n \f$ and \f$ j = 1 + (i − 1) * incx \f$ \n  
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sscal API\n";
      clk_start = clock();

      status = cublasSscal(handle, vector_length, (float *)&alpha, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Sscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sscal API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dscal API\n";
      clk_start = clock();

      status = cublasDscal(handle, vector_length, (double *)&alpha, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout<< " Dscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dscal API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cscal API\n";
      clk_start = clock();

      status = cublasCscal(handle, vector_length, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Cscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cscal API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Csscal API\n";
      clk_start = clock();

      status = cublasCsscal(handle, vector_length, (float *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Csscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Csscal API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zscal API\n";
      
      clk_start = clock();

      status = cublasZscal(handle, vector_length, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zscal API call ended\n";
      break;
    }

    case 'T': {
      std::cout << "\nCalling Zdscal API\n";
      clk_start = clock();

      status = cublasZdscal(handle, vector_length, (double *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zdscal kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zdscal API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector X after " << mode << "scal operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      break;
    }

    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      break;
    }

    case 'H': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      break;
    }

    case 'T': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
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

void mode_S(int vector_length, double alpha_real, double alpha_imaginary) {
            
  float alpha = (float)alpha_real;

  Scal<float, float> Sscal(vector_length, alpha, 'S' );
  Sscal.ScalApiCall();
}

void mode_D(int vector_length, double alpha_real, double alpha_imaginary) {
            
  double alpha = alpha_real;

  Scal<double, double> Dscal(vector_length, alpha, 'D');
  Dscal.ScalApiCall();
}

void mode_C(int vector_length, double alpha_real, double alpha_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Scal<cuComplex, cuComplex> Cscal(vector_length, alpha, 'C');
  Cscal.ScalApiCall(); 
}

void mode_H(int vector_length, double alpha_real, double alpha_imaginary) {
            
  float alpha = (float)alpha_real;

  Scal<cuComplex, float> Csscal(vector_length, alpha, 'H');
  Csscal.ScalApiCall(); 
}

void mode_Z(int vector_length, double alpha_real, double alpha_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  Scal<cuDoubleComplex, cuDoubleComplex> Zscal(vector_length, alpha, 'Z');
  Zscal.ScalApiCall(); 
}

void mode_T(int vector_length, double alpha_real, double alpha_imaginary) {
            
  double alpha = alpha_real;

  Scal<cuDoubleComplex, double> Zdscal(vector_length, alpha, 'T');
  Zdscal.ScalApiCall(); 
}


void (*cublas_func_ptr[])(int, double, double) = {
  mode_S, mode_D, mode_C, mode_H, mode_Z, mode_T
};

int main(int argc, char **argv) {

  int vector_length;
  double alpha_real, alpha_imaginary;
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  (*cublas_func_ptr[mode_index[mode]])(vector_length, alpha_real, alpha_imaginary);
  
  return 0;
}
