#include <unordered_map>
#include "cublas_RotEx_test.h"

template<class T>
RotEx<T>::RotEx(int vector_length, T sine, T cosine, char mode)
    : vector_length(vector_length), sine(sine), cosine(cosine), mode(mode) {}

template<class T>
void RotEx<T>::FreeMemory() {
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
    std::cout << " Unable to uninitialize handle \n";
  }
}

template<class T>
int RotEx<T>::RotExApiCall() {

  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostVectorX) {
    std::cout << " Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    std::cout << " Host memory allocation error (vectorY)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * If statement - To Initialize and Print input vectors based on mode passed,
   * X and Y are vectors
   */
  
  if (mode == 'S') {
    util::InitializeVector<float>((float *)HostVectorX, vector_length);
    util::InitializeVector<float>((float *)HostVectorY, vector_length);

    std::cout << "\nVector X of size " << vector_length << "\n" ;
    util::PrintVector<float>((float *)HostVectorX, vector_length);
    std::cout << "\nVector Y of size " << vector_length << "\n" ;
    util::PrintVector<float>((float *)HostVectorY, vector_length);
      
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

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY,
                           VECTOR_LEADING_DIMENSION, DeviceVectorY,
						   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector Y from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * This function is an extension to the routine cublas<t>rot where input data, output data, cosine/sine type, and compute type can be specified independently \n
   * API call to  apply Givens Rotation matrix (i.e., Rotation in the x,y plane counter-clockwise
   * by angle defined by cos(alpha) = c, sin(alpha) = s)
   * The performed operation is \f$ x[k] = c * x[k] + s * y[j] \f$ and \f$ y[j] = -s * x[k] + c * y[j] \f$
   * where \f$ k = 1 + (i - 1) * incx \f$ and \f$ j = 1 + (i - 1) * incy \f$.
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  if (mode == 'S') {
    std::cout << "\nCalling RotEx API\n";
    clk_start = clock();

    status = cublasRotEx(handle, vector_length, (float *)DeviceVectorX, CUDA_R_32F,
	                    VECTOR_LEADING_DIMENSION, (float *)DeviceVectorY, CUDA_R_32F,
                        VECTOR_LEADING_DIMENSION, (float *)&cosine, (float *)&sine,
                        CUDA_R_32F, CUDA_R_32F);

    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << " RotEx kernel execution error\n";
      FreeMemory();
      return EXIT_FAILURE;
    }

    clk_end = clock();
    std::cout << "RotEx API call ended\n";
  }

  //! Copy Vector X, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, HostVectorX,
			   VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY,
                           VECTOR_LEADING_DIMENSION, HostVectorY,
			   VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector y from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if(mode == 'S') {
    std::cout << "\nVector X after " <<  "RotEx operation is:\n";
    util::PrintVector<float>((float *)HostVectorX, vector_length);

    std::cout << "\nVector y after " << "RotEx operation is:\n";
    util::PrintVector<float>((float *)HostVectorY, vector_length);
      
  }

  long long total_operations = vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int vector_length, float sine, float cosine) {
  

  RotEx<float> SRotEx(vector_length, sine, cosine,  'S' );
  return SRotEx.RotExApiCall();
}

int (*cublas_func_ptr[])(int, float, float) = {
  mode_S
};

int main(int argc, char **argv) {
  int vector_length, status;
  float alpha, alpha_radian, sine_real;
  float cosine;
  char mode;

  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;

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

    else if (!(cmd_argument.compare("-alpha")))
      alpha = std::stof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! Check Dimension Validity
  if (vector_length <= 0){
    std::cout << "Invalid Dimension error\n";
    return EXIT_FAILURE;
  }

  alpha_radian = DEG_TO_RADIAN(alpha);
  sine_real = sin(alpha_radian);
  cosine = cos(alpha_radian);

  status = (*cublas_func_ptr[mode_index[mode]])(vector_length, sine_real, cosine);

  return status;
}
