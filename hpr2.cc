%%writefile max.cc
#include <unordered_map>
#include "hpr2.h"

template<class T>
Hpr2<T>::Hpr2(int A_row, int A_col, int vector_length, T alpha, char mode) : A_row(A_row), A_col(A_col), 
    vector_length(vector_length), alpha(alpha), mode(mode) {}

template<class T>
void Hpr2<T>::FreeMemory() {
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
int Hpr2<T>::Hpr2ApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  int matrix_size = A_row * (A_col + 1) / 2;
  HostMatrixA = new T[matrix_size];
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
   * Switch Case - To Initialize and Print input matrix and vector based on mode passed,
   * A is a hermitian packed matrix, 
   * X is a vector
   * Y is a vector
   */
  switch (mode) {

    case 'C': {
      util::InitializeSymmetricPackedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, matrix_size);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuComplex >((cuComplex *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);   
       
      break;
    }

    case 'Z': {
      util::InitializeSymmetricPackedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, matrix_size);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length); 
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);      
      
      break;
    }

  }
  
  //! Allocating Device Memory for Matrix and Vector using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceMatrixA, matrix_size * sizeof(*HostMatrixA));
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
  
  //! Copying values of Host matrix to Device matrix using cublasSetVector()
  //! Copying values of Host vector to Device vector using cublasSetVector()

  status = cublasSetVector (matrix_size, sizeof (*HostMatrixA), HostMatrixA, VECTOR_LEADING_DIMENSION, DeviceMatrixA, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying matrix A from host to device  in vector form failed\n";
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
   * API call to perform the packed Hermitian rank-2 update : \f$ A = alpha * X * Y ^ H + alpha(Imaginary part) * Y * X ^ H + A \f$
   */ 
  switch (mode) {

    case 'C': {
      std::cout << "\nCalling Chpr2 API\n";
      clk_start = clock();

      status = cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, A_row, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, 
                           VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceMatrixA);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Chpr2 kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Chpr2 API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zhpr2 API\n";
      clk_start = clock();

      status = cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, A_row, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, 
                           VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceMatrixA);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zhpr2 kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zhpr2 API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix A, holding resultant matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetVector(matrix_size, sizeof (*HostMatrixA), DeviceMatrixA, VECTOR_LEADING_DIMENSION, HostMatrixA, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output matrix A from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "hpr2 operation is:\n";

  switch (mode) {

    case 'C': {
      util::PrintSymmetricPackedUpperComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, matrix_size);
      break;
    }

    case 'Z': {
      util::PrintSymmetricPackedUpperComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, matrix_size);
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

  Hpr2<cuComplex> Chpr2(A_row, A_col, vector_length, alpha, 'C');
  Chpr2.Hpr2ApiCall(); 
}

void mode_Z(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {

  cuDoubleComplex alpha = {alpha_real, alpha_imaginary}; 
            
  Hpr2<cuDoubleComplex> Zhpr2(A_row, A_col, vector_length, alpha, 'Z');
  Zhpr2.Hpr2ApiCall(); 
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
