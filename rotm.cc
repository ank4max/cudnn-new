%%writefile test.cc
#include <unordered_map>
#include "rotm.h"

template<class T>
Rotm<T>::Rotm(int vector_length, T param_1, T param_2, T param_3, T param_4, char mode) : vector_length(vector_length), param_1(param_1),
              param_2(param_2), param_3(param_3), param_4(param_4), mode(mode) {}

template<class T>
void Rotm<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixParam)
    delete[] HostMatrixParam;

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
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Rotm<T>::RotmApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  int matrix_size = 5;
  HostMatrixParam = new T[matrix_size];
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];
  std::cout <<vector_length;

  if (!HostMatrixParam) {
    std::cout << " Host memory allocation error (matrixParam)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
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
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * Param is a matrix, 
   * X and Y are vectors
   */
  switch (mode) {

    case 'S': {
      
      util::InitializeParamMatrix<float>((float *)HostMatrixParam, param_1, param_2, param_3, param_4);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nMatrix Param of size " << "2" << " * " << "2" << ":\n";
      util::PrintParamMatrix<float>((float *)HostMatrixParam, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      
      break;
    }
    case 'D': {
      util::InitializeParamMatrix<double>((double *)HostMatrixParam, param_1, param_2, param_3, param_4);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length);

      std::cout << "\nMatrix Param of size " << "2" << " * " << "2" << ":\n";
      util::PrintParamMatrix<double>((double *)HostMatrixParam, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, vector_length);     
      
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
    std::cout << " Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorY, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying vector Y from host to device failed\n";
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
   * API call to apply the modified Givens transformation :
   * The result is \f$ x[k] = h11 × x[k] + h12 × y[j] \f$ and \f$ y[j] = h21 × x[k] + h22 × y[j] \f$ where \f$ k = 1 + (i − 1) * incx \f$
   *  and \f$ j = 1 + (i − 1) * incy \f$ . 
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   * The elements , , and of matrix H are stored in param[1], param[2], param[3] and param[4], respectively. 
   * The flag=param[0] defines the following predefined values for the matrix H entries
   */
  switch (mode) {

    case 'S': {
      std::cout << "\nCalling Srotm API\n";
      clk_start = clock();

      status = cublasSrotm(handle, vector_length, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (float *)DeviceVectorY, 
                           VECTOR_LEADING_DIMENSION, (float *)HostMatrixParam);


      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Srotm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Srotm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Drotm API\n";
      clk_start = clock();

      status = cublasDrotm(handle, vector_length, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (double *)DeviceVectorY, 
                           VECTOR_LEADING_DIMENSION, (double *)HostMatrixParam);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Drotm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Drotm API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X and  Y, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector X from device failed";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY, VECTOR_LEADING_DIMENSION, HostVectorY, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector Y from device failed";
    FreeMemory();
    return EXIT_FAILURE;
  }

  switch (mode) {

    case 'S': {
      std::cout << "\nVector X after " << mode << "rotm operation is:\n";
      util::PrintVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rotm operation is:\n";
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      std::cout << "\nVector X after " << mode << "rotm operation is:\n";
      util::PrintVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rotm operation is:\n";
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;
    }
  }

  long long total_operations = (matrix_size - 1) * vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}


void mode_S(int vector_length, double param_1, double param_2,
            double param_3, double param_4) {
            
  float param_f1 = (float)param_1;
  float param_f2 = (float)param_2;
  float param_f3 = (float)param_3;
  float param_f4 = (float)param_4;

  Rotm<float> Srotm(vector_length, param_f1, param_f2, param_f3, param_f4, 'S');
  Srotm.RotmApiCall(); 
}

void mode_D(int vector_length, double param_1, double param_2,
            double param_3, double param_4) {
            

  Rotm<double> Drotm(vector_length, param_1, param_2, param_3, param_4, 'D');
  Drotm.RotmApiCall(); 
}

void (*cublas_func_ptr[])(int, double, double, double, double) = {
 mode_S, mode_D
};

int main(int argc, char **argv) {
  int vector_length;
  double param_1, param_2, param_3, param_4;
  char mode;

  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;

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

   else if (!(cmd_argument.compare("-param_1")))
      param_1 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-param_2")))
      param_2 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-param_3")))
      param_3 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-param_4")))
      param_4 = std::stod(argv[loop_count + 1]);   

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  (*cublas_func_ptr[mode_index[mode]])(vector_length, param_1, param_2, 
                                       param_3, param_4);

  return 0;
}
