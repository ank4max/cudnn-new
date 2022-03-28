#include <unordered_map>
#include "cublas_symv_test.h"

template<class T>
Symv<T>::Symv(int A_row, int A_col, int vector_length, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length),
      alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Symv<T>::FreeMemory() {
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
    std::cout << "!!!! Unable to uninitialize handle " << std::endl;
  }
}

template<class T>
int Symv<T>::SymvApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostMatrixA) {
    std::cout << "!!!! Host memory allocation error (matrixA)" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    std::cout << "!!!! Host memory allocation error (vectorY)" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A is a symmetric matrix, X and Y are vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);

      break;
    }

    case 'D': {
      util::InitializeSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, vector_length);

      break;
    }

    case 'C': {
      util::InitializeSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

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

  cudaStatus = cudaMalloc((void **)&DeviceVectorY, vector_length * sizeof(*HostVectorY));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Initializing CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host matrix to Device matrices using cublasSetMatrix()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row,
                           DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix A from host to device failed" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX,
                           VECTOR_LEADING_DIMENSION, DeviceVectorX,
			   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY,
                           VECTOR_LEADING_DIMENSION, DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector Y from host to device failed" << std::endl;
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
   * API call to perform the symmetric matrix-vector multiplication.: \f$ Y = alpha * A * X + beta * Y \f$
   * This function has an alternate faster implementation using atomics that can be enabled with cublasSetAtomicsMode().
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Ssymv API\n";
      clk_start = clock();

      status = cublasSsymv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
		           (float *)&alpha, (float *)DeviceMatrixA, A_row,
                           (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (float *)&beta, (float *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ssymv kernel execution error" << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ssymv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dsymv API\n";
      clk_start = clock();

      status = cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
	                   (double *)&alpha, (double *)DeviceMatrixA, A_row,
                           (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (double *)&beta, (double *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Dsymv kernel execution error" << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dsymv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Csymv API\n";
      clk_start = clock();

      status = cublasCsymv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
	                   (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (cuComplex *)&beta, (cuComplex *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Csymv kernel execution error" << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Csymv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zsymv API\n";
      clk_start = clock();

      status = cublasZsymv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
	                   (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA,
			   A_row, (cuDoubleComplex *)DeviceVectorX,
			   VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)&beta,
			   (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zsymv kernel execution error" << std::endl;
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zsymv API call ended\n";
      break;
    }
  }

  //! Copy Vector Y, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY,
                           VECTOR_LEADING_DIMENSION, HostVectorY,
			   VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output vector Y from device" << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "symv operation is:\n";

  switch (mode) {
    case 'S': {
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;
    }

    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
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

int mode_S(int A_row, int A_col, int vector_length, double alpha_real,
            double alpha_imaginary, double beta_real, double beta_imaginary) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Symv<float> Ssymv(A_row, A_col, vector_length, alpha, beta, 'S');
  return Ssymv.SymvApiCall();
}

int mode_D(int A_row, int A_col, int vector_length, double alpha_real,
            double alpha_imaginary, double beta_real, double beta_imaginary) {
  double alpha = alpha_real;
  double beta = beta_real;

  Symv<double> Dsymv(A_row, A_col, vector_length, alpha, beta, 'D');
  return Dsymv.SymvApiCall();
}

int mode_C(int A_row, int A_col, int vector_length, double alpha_real,
            double alpha_imaginary, double beta_real, double beta_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Symv<cuComplex> Csymv(A_row, A_col, vector_length, alpha, beta, 'C');
  return Csymv.SymvApiCall();
}

int mode_Z(int A_row, int A_col, int vector_length, double alpha_real,
            double alpha_imaginary, double beta_real, double beta_imaginary) {
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Symv<cuDoubleComplex> Zsymv(A_row, A_col, vector_length, alpha, beta, 'Z');
  return Zsymv.SymvApiCall();
}

int (*cublas_func_ptr[])(int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
 
  //! Dimension check
  if (A_row <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  //! initializing values for A column and vector size
  A_col = A_row;
  vector_length = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, alpha_real,
                                       alpha_imaginary, beta_real, beta_imaginary);
  
  std::cout <<status;

  return status;
}
