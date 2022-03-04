%%writefile max.cc
#include <unordered_map>
#include "Swap.h"

template<class T>
Swap<T>::Swap(int vector_length, char mode)
              : vector_length(vector_length), mode(mode) {}

template<class T>
void Swap<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Swap<T>::SwapApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostVectorX) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorX)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorY)\n");
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
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector Y from host to device failed\n");
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
   * API call to interchange the elements of vector x and y \n
   * The performed operation is \f$ y[j] ⇔ x[k] \f$ for i = 1,…,n where \f$ k = 1 + (i − 1) * incx \f$ and 
   * \f$ j = 1 + (i − 1) * incy \f$
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sswap API\n";
      clk_start = clock();

      status = cublasSswap(handle, vector_length, (float *)DeviceVectorX, 1, (float *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sswap kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sswap API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dswap API\n";
      clk_start = clock();

      status = cublasDswap(handle, vector_length, (double *)DeviceVectorX, 1, (double *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dswap kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dswap API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cswap API\n";
      clk_start = clock();

      status = cublasCswap(handle, vector_length, (cuComplex *)DeviceVectorX, 1, (cuComplex *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cswap kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cswap API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zswap API\n";
      clk_start = clock();

      status = cublasZswap(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1, (cuDoubleComplex *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zswap kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zswap API call ended\n";
      break;
    }
  }
  
  //! Copy Vectors X and Y, holding resultant Vectors, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, 1, HostVectorX, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector x from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY, 1, HostVectorY, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector y from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }


  switch (mode) {
    case 'S': { 

      std::cout << "\nVectors X after " << mode << "swap operation is:\n";
      util::PrintVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nVectors Y after " << mode << "swap operation is:\n";
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      
      std::cout << "\nVectors X after " << mode << "swap operation is:\n"; 
      util::PrintVector<double>((double *)HostVectorX, vector_length); 

      std::cout << "\nVectors Y after " << mode << "swap operation is:\n";
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;
    }

    case 'C': {

      std::cout << "\nVectors X after " << mode << "swap operation is:\n"; 
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nVectors Y after " << mode << "swap operation is:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      break;
    }

    case 'Z': {

      std::cout << "\nVectors X after " << mode << "swap operation is:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nVectors Y after " << mode << "swap operation is:\n";
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

void mode_S(int vector_length) {           

  Swap<float> Sswap(vector_length, 'S' );
  Sswap.SwapApiCall();
}

void mode_D(int vector_length) {

  Swap<double> Dswap(vector_length, 'D');
  Dswap.SwapApiCall();
}

void mode_C(int vector_length) {

  Swap<cuComplex> Cswap(vector_length, 'C');
  Cswap.SwapApiCall(); 
}

void mode_Z(int vector_length) {

  Swap<cuDoubleComplex> Zswap(vector_length, 'Z');
  Zswap.SwapApiCall(); 
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
