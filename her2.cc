%%writefile test.cc
#include <unordered_map>
#include "her2.h"

template<class T>
Her2<T>::Her2(int A_row, int A_col, int vector_length, T alpha, char mode) : A_row(A_row), A_col(A_col), 
    vector_length(vector_length), alpha(alpha), mode(mode) {}

template<class T>
void Her2<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostVectorX)
    delete[] HostVectorX;

  if (HostVectorY)
    delete[] HostVectorY;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

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
int Her2<T>::Her2ApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostMatrixA) {
    std::cout << " Host memory allocation error (matrixA)\n";
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
   * A is a hermitian matrix, 
   * X is a vector
   * Y is a vector
   */
  switch (mode) {

    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);   
       
      break;
    }

    case 'Z': {
      util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length); 
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);      
      
      break;
    }

  }
  
  //! Allocating Device Memory for Matrix and Vector using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceMatrixA, A_row * A_col * sizeof(*HostMatrixA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

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
  
  //! Copying values of Host matrix to Device matrix using cublasSetMatrix()
  //! Copying values of Host vector to Device vector using cublasSetVector()

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying matrix A from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

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
   * The Error values returned by API are : 
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully 
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized 
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n<0 or incx,incy=0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU 
   */
    
  /**
   * API call to perform the Hermitian rank-2 update : \f$ A = alpha * X * Y ^ H + alpha(Imaginary part) * Y * X ^ H + A \f$
   */ 
  switch (mode) {

    case 'C': {
      std::cout << "\nCalling Cher2 API\n";
      clk_start = clock();

      status = cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, A_row, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, 
                           VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Cher2 kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cher2 API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zher2 API\n";
      clk_start = clock();

      status = cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, A_row, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, 
                           VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zher2 kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zher2 API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix A, holding resultant matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(A_row, A_col, sizeof(*HostMatrixA),
                           DeviceMatrixA, A_row, HostMatrixA, A_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output matrix A from device failed";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "her2 operation is:\n";

  switch (mode) {

    case 'C': {
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'Z': {
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      break;
    }
  }

  long long total_operations = A_row * vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}


void mode_C(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {

  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Her2<cuComplex> Cher2(A_row, A_col, vector_length, alpha, 'C');
  Cher2.Her2ApiCall(); 
}

void mode_Z(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {

  cuDoubleComplex alpha = {alpha_real, alpha_imaginary}; 
            
  Her2<cuDoubleComplex> Zher2(A_row, A_col, vector_length, alpha, 'Z');
  Zher2.Her2ApiCall(); 
}

void (*cublas_func_ptr[])(int, int, int, double, double) = {
 mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length;
  double alpha_real, alpha_imaginary;
  char mode;

  std::unordered_map<char, int> mode_index;
  mode_index['C'] = 0;
  mode_index['Z'] = 1;

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

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

   else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

   else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

   //! Initializing values for A column  and vector size
   A_col = A_row;
   vector_length = A_row;


  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, alpha_real, alpha_imaginary);

  return 0;
}
