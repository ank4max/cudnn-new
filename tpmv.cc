%%writefile mx.cc
#include <unordered_map>
#include "tpmv.h"

template<class T>
Tpmv<T>::Tpmv(int A_row, int A_col, int vector_length, char mode) : A_row(A_row), A_col(A_col), 
                                                                    vector_length(vector_length), mode(mode) {}

template<class T>
void Tpmv<T>::FreeMemory() {
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
int Tpmv<T>::TpmvApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
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
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A is a triangular packed matrix, 
   * X is a vector
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
      util::InitializeVector<double >((double *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<double >((double *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double >((double  *)HostVectorX, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeSymmetricPackedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, matrix_size);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuComplex >((cuComplex *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);  
       
      break;
    }

    case 'Z': {
      util::InitializeSymmetricPackedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, matrix_size);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, matrix_size);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);      
      
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
  
  //! Copying values of Host matrix to Device matrix using cublasSetVector()
  //! Copying values of Host vectors to Device vectors using cublasSetVector()

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
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, or incx=0 \n
   * CUBLAS_STATUS_ALLOC_FAILED - The allocation of internal scratch memory failed \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to perform the triangular packed matrix-vector multiplication : \f$ X = op(A) * X \f$ 
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Stpmv API\n";
      clk_start = clock();

      status = cublasStpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                           A_row, (float *)DeviceMatrixA, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Stpmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Stpmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtpmv API\n";
      clk_start = clock();

      status = cublasDtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                           A_row, (double *)DeviceMatrixA, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dtpmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtpmv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctpmv API\n";
      clk_start = clock();

      status = cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                           A_row, (cuComplex *)DeviceMatrixA, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ctpmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctpmv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztpmv API\n";
      clk_start = clock();

      status = cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                           A_row, (cuDoubleComplex *)DeviceMatrixA, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ztpmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztpmv API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector X from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector X after " << mode << "tpmv operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      break;
    }

    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
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

void mode_S(int A_row, int A_col, int vector_length) {

  Tpmv<float> Stpmv(A_row, A_col, vector_length, 'S' );
  Stpmv.TpmvApiCall();
}

void mode_D(int A_row, int A_col, int vector_length) {
            
  Tpmv<double> Dtpmv(A_row, A_col, vector_length, 'D');
  Dtpmv.TpmvApiCall();
}

void mode_C(int A_row, int A_col, int vector_length) {
            
  Tpmv<cuComplex> Ctpmv(A_row, A_col, vector_length, 'C');
  Ctpmv.TpmvApiCall();
}

void mode_Z(int A_row, int A_col, int vector_length) {
            
  Tpmv<cuDoubleComplex> Ztpmv(A_row, A_col, vector_length, 'Z');
  Ztpmv.TpmvApiCall();
}

void (*cublas_func_ptr[])(int, int, int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length;
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

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  //! initializing values for A column and vector size
  A_col = A_row;
  vector_length = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length);

  return 0;
}
