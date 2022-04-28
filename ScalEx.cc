#include "cublas_ScalEx_test.h"

template<class T>
ScalEx<T>::ScalEx(int vector_length, T alpha)
              : vector_length(vector_length), alpha(alpha) {}

template<class T>
void ScalEx<T>::FreeMemory() {
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

template<class T>
int ScalEx<T>::ScalExApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];

  if (!HostVectorX) {
    std::cout << " Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Initializing and Printing Vector X
  util::InitializeVector<float>((float *)HostVectorX, vector_length);

  std::cout << "\nVector X of size " << vector_length << "\n" ;
  util::PrintVector<float>((float *)HostVectorX, vector_length);
  

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
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX,
                           VECTOR_LEADING_DIMENSION, DeviceVectorX,
						   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * API call to ScalExe the vector x by the ScalExar α and overwrites it with the resul: \f$ X = alpha * X \f$ \n
   * The performed operation is \f$ x[j] = α* x[j] for i = 1, ..., n \f$ and \f$ j = 1 + (i - 1) * incx \f$ \n
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_NOT_SUPPORTED - The combination of the parameters xType and executionType is not supported\n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  
  std::cout << "\nCalling ScalEx API\n";
  clk_start = clock();

  status = cublasScalEx(handle, vector_length, (float *)&alpha, CUDA_R_32F,
	               (float *)DeviceVectorX, CUDA_R_32F, VECTOR_LEADING_DIMENSION, CUDA_R_32F);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " ScalEx kernel execution error\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  clk_end = clock();
  std::cout << "ScalEx API call ended\n";

  //! Copy Vector X, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX),
                           DeviceVectorX, VECTOR_LEADING_DIMENSION,
		           HostVectorX, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector X after " << "ScalEx operation is:\n";
  
  //! Printing Output Vector
  util::PrintVector<float>((float *)HostVectorX, vector_length);

  long long total_operations = vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  int vector_length, status;
  float alpha_real;
  
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stof(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (vector_length <= 0){
    std::cout << "Invalid Dimension error\n";
    return EXIT_FAILURE;
  }

  ScalEx<float> scalEx(vector_length, alpha_real);
  status = scalEx.ScalExApiCall();
  return status;
}
