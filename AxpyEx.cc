%%writefile mex.cc
#include <unordered_map>
#include "cublas_AxpyEx_test.h"

template<class T>
AxpyEx<T>::AxpyEx(int vector_length, T alpha, char mode)
    : vector_length(vector_length), alpha(alpha), mode(mode) {}

template<class T>
void AxpyEx<T>::FreeMemory() {
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
int AxpyEx<T>::AxpyExApiCall() {
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

    case 'H': {
      util::InitializeVector<__half>((__half *)HostVectorX, vector_length);
      util::InitializeVector<__half>((__half *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<__half>((__half *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<__half>((__half *)HostVectorY, vector_length);
          
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
   * API call to multiply the vector x by the scalar α and adds it to the vector y : \f$ Y = alpha * X + Y \f$ \n
   * This function is an API generalization of the routine cublas<t>AxpyEx where input data, output data and compute type can be specified 
     independently.
   * Hence, the performed operation is \f$ y[j] = α × x[k] + y[j] for i=1,…,n \f$ where \f$ k = 1 + (i − 1) * incx \f$ and 
   * \f$ j = 1 + (i − 1) * incy \f$ \n . 
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */

  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SAxpyEx API\n";
      clk_start = clock();

      status = cublasAxpyEx(handle, vector_length, (float *)&alpha, CUDA_R_32F, (float *)DeviceVectorX, CUDA_R_32F, 
                            VECTOR_LEADING_DIMENSION, (float *)DeviceVectorY, CUDA_R_32F, VECTOR_LEADING_DIMENSION, CUDA_R_32F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  SAxpyEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SAxpyEx API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DAxpyEx API\n";
      clk_start = clock();

      status = cublasAxpyEx(handle, vector_length, (double *)&alpha, CUDA_R_64F, (double *)DeviceVectorX, CUDA_R_64F, 
                            VECTOR_LEADING_DIMENSION, (double *)DeviceVectorY, CUDA_R_64F, VECTOR_LEADING_DIMENSION, CUDA_R_64F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  DAxpyEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DAxpyEx API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CAxpyEx API\n";
      clk_start = clock();

      status = cublasAxpyEx(handle, vector_length, (cuComplex *)&alpha, CUDA_C_32F, (cuComplex *)DeviceVectorX, CUDA_C_32F, 
                            VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceVectorY, CUDA_C_32F, VECTOR_LEADING_DIMENSION, CUDA_C_32F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  CAxpyEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CAxpyEx API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZAxpyEx API\n";
      clk_start = clock();

      status = cublasAxpyEx(handle, vector_length, (cuDoubleComplex *)&alpha, CUDA_C_64F, (cuDoubleComplex *)DeviceVectorX, CUDA_C_64F, 
                            VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceVectorY, CUDA_C_64F, VECTOR_LEADING_DIMENSION, CUDA_C_64F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  ZAxpyEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZAxpyEx API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling HAxpyEx API\n";
      clk_start = clock();

      status = cublasAxpyEx(handle, vector_length, (__half *)&alpha, CUDA_R_16F, (__half *)DeviceVectorX, CUDA_R_16F, 
                            VECTOR_LEADING_DIMENSION, (__half *)DeviceVectorY, CUDA_R_16F, VECTOR_LEADING_DIMENSION, CUDA_R_32F);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  HAxpyEx kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "HAxpyEx API call ended\n";
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

  std::cout << "\nVector Y after " << mode << "AxpyEx operation is:\n";

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

    case 'H': {  
      util::PrintVector<__half>((__half *)HostVectorY, vector_length);
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

  AxpyEx<float> SAxpyEx(vector_length, alpha, 'S');

  return SAxpyEx.AxpyExApiCall();
}

int mode_D(int vector_length, double alpha_real, double alpha_imaginary) {
  double alpha = alpha_real;

  AxpyEx<double> DAxpyEx(vector_length, alpha, 'D');

  return DAxpyEx.AxpyExApiCall();
}

int mode_C(int vector_length, double alpha_real, double alpha_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  AxpyEx<cuComplex> CAxpyEx(vector_length, alpha, 'C');

  return CAxpyEx.AxpyExApiCall(); 
}

int mode_Z(int vector_length, double alpha_real, double alpha_imaginary) {
    cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  AxpyEx<cuDoubleComplex> ZAxpyEx(vector_length, alpha, 'Z');

  return ZAxpyEx.AxpyExApiCall(); 
}

int mode_H(int vector_length, double alpha_real, double alpha_imaginary) {    
  __half alpha = (__half)alpha_real;

  AxpyEx<__half> HAxpyEx(vector_length, alpha, 'H');

  return HAxpyEx.AxpyExApiCall();
}


int (*cublas_func_ptr[])(int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z, mode_H
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (vector_length <= 0) {
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }
 
  status = (*cublas_func_ptr[mode_index[mode]])(vector_length, alpha_real, alpha_imaginary);
  
  return status;
}
