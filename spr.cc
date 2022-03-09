%%writefile max.cc
#include <unordered_map>
#include "spr.h"

template<class T>
Spr<T>::Spr(int A_row, int A_col, int vector_length, T alpha, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length), 
      alpha(alpha), mode(mode) {}

template<class T>
void Spr<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostVectorX)
    delete[] HostVectorX;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorX);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Spr<T>::SprApiCall() {
  //! Allocating Host Memory for Matrix and Vector

  int matrix_size = A_row * (A_col + 1) / 2;
  HostMatrixA = new T[matrix_size];
  HostVectorX = new T[vector_length];

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


  /**
   * Switch Case - To Initialize and Print input matrix and vector based on mode passed,
   * A is a symmetric packed matrix, X is a vector
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricPackedMatrix<float>((float *)HostMatrixA, matrix_size);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<float>((float *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeSymmetricPackedMatrix<double>((double *)HostMatrixA, matrix_size);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<double>((double *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
       
      break;
    }
  }
  
  //! Allocating Device Memory for Matrix and Vectors using cudaMalloc()
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

  //! Initializing CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
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

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - the parameters n < 0 or incx, incy = 0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to perform the packed symmetric rank-1 update : \f$ A = alpha * X * X ^ T + A \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sspr API\n";
      clk_start = clock();

      status = cublasSspr(handle, CUBLAS_FILL_MODE_LOWER, A_row, (float *)&alpha, (float *)DeviceVectorX, 
                          VECTOR_LEADING_DIMENSION, (float *)DeviceMatrixA);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Sspr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sspr API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dspr API\n";
      clk_start = clock();

      status = cublasDspr(handle, CUBLAS_FILL_MODE_LOWER, A_row, (double *)&alpha, (double *)DeviceVectorX, 
                          VECTOR_LEADING_DIMENSION, (double *)DeviceMatrixA);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dspr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dspr API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix A, holding resultant Matrix, from Device to Host using cublasGetVector()
  status = cublasGetVector(matrix_size, sizeof (*HostMatrixA), DeviceMatrixA, VECTOR_LEADING_DIMENSION, HostMatrixA, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output matrix A from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "spr operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintSymmetricPackedUpperMatrix<float>((float *)HostMatrixA, A_row, matrix_size);
      break;
    }

    case 'D': {
      util::PrintSymmetricPackedUpperMatrix<double>((double *)HostMatrixA, A_row, matrix_size);
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

void mode_S(int A_row, int A_col, int vector_length, double alpha_real) {
  float alpha = (float)alpha_real;

  Spr<float> Sspr(A_row, A_col, vector_length, alpha, 'S');
  Sspr.SprApiCall();
}

void mode_D(int A_row, int A_col, int vector_length, double alpha_real) {   
  double alpha = alpha_real;

  Spr<double> Dspr(A_row, A_col, vector_length, alpha, 'D');
  Dspr.SprApiCall();
}


void (*cublas_func_ptr[])(int, int, int, double) = {
  mode_S, mode_D
};

int main(int argc, char **argv) {

  int A_row, A_col, vector_length;
  double alpha_real;
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

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! initializing values for A column and vector size
  A_col = A_row;
  vector_length = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, alpha_real);
  
  return 0;
}
