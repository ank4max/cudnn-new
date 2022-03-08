%%writefile ma.cc
#include <unordered_map>
#include "trmv.h"

template<class T>
Trmv<T>::Trmv(int A_row, int A_col, int vector_length, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length), 
      mode(mode) {}

template<class T>
void Trmv<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Trmv<T>::TrmvApiCall() {
  //! Allocating Host Memory for Matrix and Vector
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[vector_length];

  if (!HostMatrixA) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorX) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorX)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrix and vector based on mode passed,
   * A is a triangular matrix, X is a vector
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      
      break;
    }

    case 'Z': {
      util::InitializeSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);     
      
      break;
    }

  }
  
  //! Allocating Device Memory for Matrix and Vectors using cudaMalloc()
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

  //! Initializing CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host matrix to Device matriX using cublasSetMatrix()
  //! Copying values of Host vector to Device vector using cublasSetVector()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  } 

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - the parameters m, n < 0 or incx, incy = 0
   * CUBLAS_STATUS_ALLOC_FAILED - The allocation of internal scratch memory failed \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to performs the triangular matrix-vector multiplication : \f$ X = op(A) * X \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Strmv API\n";
      clk_start = clock();

      status = cublasStrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, vector_length,
                           (float *)DeviceMatrixA, A_row, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Strmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Strmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtrmv API\n";
      clk_start = clock();

      status = cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, vector_length,
                           (double *)DeviceMatrixA, A_row, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dtrmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtrmv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctrmv API\n";
      clk_start = clock();

      status = cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, vector_length,
                           (cuComplex *)DeviceMatrixA, A_row, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ctrmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctrmv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztrmv API\n";
      clk_start = clock();

      status = cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, vector_length,
                           (cuDoubleComplex *)DeviceMatrixA, A_row, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ztrmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztrmv API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, 
                           VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector y from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector X after " << mode << "trmv operation is:\n";

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

  Trmv<float> Strmv(A_row, A_col, vector_length, 'S');
  Strmv.TrmvApiCall();
}

void mode_D(int A_row, int A_col, int vector_length) {   

  Trmv<double> Dtrmv(A_row, A_col, vector_length, 'D');
  Dtrmv.TrmvApiCall();
}

void mode_C(int A_row, int A_col, int vector_length) {

  Trmv<cuComplex> Ctrmv(A_row, A_col, vector_length, 'C');
  Ctrmv.TrmvApiCall(); 
}

void mode_Z(int A_row, int A_col, int vector_length) {

  Trmv<cuDoubleComplex> Ztrmv(A_row, A_col, vector_length, 'Z');
  Ztrmv.TrmvApiCall(); 
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
