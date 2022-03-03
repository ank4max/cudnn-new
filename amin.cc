%%writefile max.cc
#include <unordered_map>
#include "max.h"

template<class T>
Amin<T>::Amin(int x_size, char mode)
              : x_size(x_size), mode(mode) {}

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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Amin<T>::AminApiCall() {
  //! Allocating Host Memory for Vector
  HostVectorX = new T[x_size];

  if (!HostVectorX) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorX)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vector based on mode passed,
   * X is a general vector 
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeVector<float>((float *)HostVectorX, x_size);
  
      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintVector<float>((float *)HostVectorX, x_size);   
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, x_size);

      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintVector<double>((double *)HostVectorX, x_size);  
      break;
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, x_size);

      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, x_size);     
      break;
    }

    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, x_size);

      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, x_size);     
      break;
    }

  }
  
  //! Allocating Device Memory for Vector using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceVectorX, x_size * sizeof(*HostVectorX));
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
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(x_size, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to finds the (smallest) index of the element of the minimum magnitude
   */
    
  /**
   * The Error values returned by API are : 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully 
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized 
   * CUBLAS_STATUS_ALLOC_FAILED - the reduction buffer could not be allocated
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU 
   */

  int result;
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling IsAmin API\n";
      clk_start = clock();

      status = cublasIsamin(handle, x_size, (float *)DeviceVectorX, 1, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  IsAmin kernel execution error\n");
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

      status = cublasIdamin(handle, x_size, (double *)DeviceVectorX, 1, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  IdAmin kernel execution error\n");
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

      status = cublasIcamin(handle, x_size, (cuComplex *)DeviceVectorX, 1, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  IcAmin kernel execution error\n");
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

      status = cublasIzamin(handle, x_size, (cuDoubleComplex *)DeviceVectorX, 1, &result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  IzAmin kernel execution error\n");
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

  switch (mode) {
    case 'S': {  
      util::PrintAmin<float>((float *)HostVectorX, result);
      break;
    }

    case 'D': {
      util::PrintAmin<double>((double *)HostVectorX, result);
      break;
    }

    case 'C': {
      util::PrintComplexAmin<cuComplex>((cuComplex *)HostVectorX, result);
      break;
    }

    case 'Z': {
      util::PrintComplexAmin<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, result);
      break;
    }  
    
  }

  long long total_operations =  x_size;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(int x_size) {
            
  Amin<float> SAmin(x_size, 'S' );
  SAmin.AminApiCall();
}

void mode_D(int x_size) {
            
  Amin<double> DAmin(x_size, 'D');
  DAmin.AminApiCall();
}

void mode_C(int x_size) {            

  Amin<cuComplex> CAmin(x_size, 'C');
  CAmin.AminApiCall(); 
}

void mode_Z(int x_size) {
 
  Amin<cuDoubleComplex> ZAmin(x_size, 'Z');
  ZAmin.AminApiCall(); 
}


void (*cublas_func_ptr[])(int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int x_size;
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

    if (!(cmd_argument.compare("-x_size")))
      x_size = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  (*cublas_func_ptr[mode_index[mode]])(x_size);
  
  return 0;
}
