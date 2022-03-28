#include <unordered_map>
#include "cublas_axpy_test.h"

template<class T>
Axpy<T>::Axpy(int vector_length, T alpha, char mode)
    : vector_length(vector_length), alpha(alpha), mode(mode) {}

template<class T>
void Axpy<T>::FreeMemory() {
  //! Free Host Memory
  if (HostVectorX)
    delete[] HostVectorX;

  if (HostVectorY)
    delete[] HostVectorY;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceVectorX);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorY);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for Y" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Axpy<T>::AxpyApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    std::cout << "!!!! Host memory allocation error (vectorY)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vectors based on mode passed,
   * X and Y are vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
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
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, VECTOR_LEADING_DIMENSION, DeviceVectorX, 
                           VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, VECTOR_LEADING_DIMENSION, DeviceVectorY,
                           VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector Y from host to device failed\n";
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
   * API call to multiply the vector x by the scalar α and adds it to the vector y : \f$ Y = alpha * X + Y \f$ \n
   * Hence, the performed operation is \f$ y[j] = α × x[k] + y[j] for i=1,…,n \f$ where \f$ k = 1 + (i − 1) * incx \f$ and 
   * \f$ j = 1 + (i − 1) * incy \f$ \n . 
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Saxpy API\n";
      clk_start = clock();

      status = cublasSaxpy(handle, vector_length, (float *)&alpha, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION, 
               (float *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Saxpy kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Saxpy API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Daxpy API\n";
      clk_start = clock();

      status = cublasDaxpy(handle, vector_length, (double *)&alpha, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION, 
                          (double *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Daxpy kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Daxpy API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Caxpy API\n";
      clk_start = clock();

      status = cublasCaxpy(handle, vector_length, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, 
                           (cuComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Caxpy kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Caxpy API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zaxpy API\n";
      clk_start = clock();

      status = cublasZaxpy(handle, vector_length, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, 
                          (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zaxpy kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zaxpy API call ended\n";
      break;
    }
  }
  
  //! Copy Vector Y, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY, VECTOR_LEADING_DIMENSION, HostVectorY, 
                           VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output vector y from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "axpy operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;
    }

    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
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

int mode_S(int vector_length, double alpha_real, double alpha_imaginary) {    
  float alpha = (float)alpha_real;

  Axpy<float> Saxpy(vector_length, alpha, 'S' );
  return Saxpy.AxpyApiCall();
}

int mode_D(int vector_length, double alpha_real, double alpha_imaginary) {
  double alpha = alpha_real;

  Axpy<double> Daxpy(vector_length, alpha, 'D');
  return Daxpy.AxpyApiCall();
}

int mode_C(int vector_length, double alpha_real, double alpha_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Axpy<cuComplex> Caxpy(vector_length, alpha, 'C');
  return Caxpy.AxpyApiCall(); 
}

int mode_Z(int vector_length, double alpha_real, double alpha_imaginary) {
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  Axpy<cuDoubleComplex> Zaxpy(vector_length, alpha, 'Z');
  return Zaxpy.AxpyApiCall(); 
}


int (*cublas_func_ptr[])(int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int vector_length, status;
  double alpha_real, alpha_imaginary;
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Dimension check
  if (vector_length <= 0){
    std::cout << "Minimum Dimension error\n";
    return EXIT_FAILURE;
  }
  
  status = (*cublas_func_ptr[mode_index[mode]])(vector_length, alpha_real, alpha_imaginary);
  
  return status;
}
