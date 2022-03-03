%%writefile sum.cc
#include <unordered_map>
#include "sum.h"

template<class T>
Asum<T>::Asum(int vector_length, char mode)
              : vector_length(vector_length), mode(mode) {}

template<class T>
void Asum<T>::FreeMemory() {
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
int Asum<T>::AsumApiCall() {
  //! Allocating Host Memory for Vector
  HostVectorX = new T[vector_length];

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
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
  
      std::cout << "\nVector X of size " << vector_length << " * 1 : \n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);   
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << " * 1 : \n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);  
      break;
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << " * 1 : \n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);     
      break;
    }

    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nVector X of size " << vector_length << " * 1 : \n" ;
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
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to computes the sum of the absolute values of the elements of vector x
   */
    
  /**
   * The Error values returned by API are : 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully 
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized 
   * CUBLAS_STATUS_ALLOC_FAILED - the reduction buffer could not be allocated
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU 
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sasum API\n";
      float result;
      clk_start = clock();

      status = cublasSasum(handle, vector_length, (float *)DeviceVectorX, 1, (float *)&result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sasum kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sasum API call ended\n";
      std::cout << "Sum of the absolute values of elements of vector X  after " << mode << "asum operation : " << abs((result));
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dasum API\n";
      double result;
      clk_start = clock();

      status = cublasDasum(handle, vector_length, (double *)DeviceVectorX, 1, (double *)&result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dasum kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dasum API call ended\n";
      std::cout << "Sum of the absolute values of elements of vector X  after " << mode << "asum operation : " << abs((result));
      break;
    }

    case 'C': {
      std::cout << "\nCalling Scasum API\n";
      float result;
      clk_start = clock();

      status = cublasScasum(handle, vector_length, (cuComplex *)DeviceVectorX, 1, (float *)&result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Scasum kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Scasum API call ended\n";
      std::cout << "Sum of the absolute values of elements of vector X  after " << mode << "asum operation : " << abs((result));
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Dzasum API\n";
      double result;
      clk_start = clock();

      status = cublasDzasum(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, 1, (double *)&result);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dzasum kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DzasumAPI call ended\n";
      std::cout << "Sum of the absolute values of elements of vector X  after " << mode << "asum operation : " << abs((result));
      break;
    }
  }


  long long total_operations =  vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(int vector_length) {
            
  Asum<float> SAsum(vector_length, 'S' );
  SAsum.AsumApiCall();
}

void mode_D(int vector_length) {
            
  Asum<double> DAsum(vector_length, 'D');
  DAsum.AsumApiCall();
}

void mode_C(int vector_length) {            

  Asum<cuComplex> CAsum(vector_length, 'C');
  CAsum.AsumApiCall(); 
}

void mode_Z(int vector_length) {
 
  Asum<cuDoubleComplex> ZAsum(vector_length, 'Z');
  ZAsum.AsumApiCall(); 
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
