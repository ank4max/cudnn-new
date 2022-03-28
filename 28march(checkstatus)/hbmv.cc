#include <unordered_map>
#include "cublas_hbmv_test.h"

template<class T>
Hbmv<T>::Hbmv(int A_row, int A_col, int vector_length, int sub_diagonals,
	T alpha, T beta, char mode) : A_row(A_row), A_col(A_col),
	vector_length(vector_length), sub_diagonals(sub_diagonals), alpha(alpha),
	beta(beta), mode(mode) {}

template<class T>
void Hbmv<T>::FreeMemory() {
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
int Hbmv<T>::HbmvApiCall() {
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
   * A is Hermitian banded matrix,
   * X and Y are vectors
   */
  switch (mode) {
    case 'C': {
      util::InitializeTriangularBandedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedComplexMatrix<cuComplex >((cuComplex *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      break;
    }
    case 'Z': {
      util::InitializeTriangularBandedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row);
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
    std::cout << " Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Copying values of Host matrix to Device matrix using cublasSetMatrix()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying matrix A from host to device failed\n";
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
   * The Error values returned by API are :
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n<0 or incx,incy=0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU
   */

  /**
   * API call to performs the Hermitian banded matrix-vector multiplication : \f$ Y = alpha * A * X + beta * Y \f$
   */
  switch (mode) {
    case 'C': {
      std::cout << "\nCalling Chbmv API\n";
      clk_start = clock();

      status = cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, A_row, sub_diagonals,
	                   (cuComplex *)&alpha, (cuComplex *)DeviceMatrixA, A_row,
                           (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (cuComplex *)&beta, (cuComplex *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Chbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Chbmv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zhbmv API\n";
      clk_start = clock();

      status = cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, A_row, sub_diagonals,
		           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA,
			   A_row, (cuDoubleComplex *)DeviceVectorX,
			   VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)&beta,
			   (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zhbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zhbmv API call ended\n";
      break;
    }
  }

  //! Copy Vector Y, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY,
                           VECTOR_LEADING_DIMENSION, HostVectorY,
			   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector Y from device failed";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "hbmv operation is:\n";
  switch (mode) {
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

int mode_C(int A_row, int A_col, int vector_length, int sub_diagonals,
	        double alpha_real, double alpha_imaginary, double beta_real,
			double beta_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Hbmv<cuComplex> Chbmv(A_row, A_col, vector_length, sub_diagonals, alpha, beta, 'C');
  return Chbmv.HbmvApiCall();
}

int mode_Z(int A_row, int A_col, int vector_length, int sub_diagonals,
	        double alpha_real, double alpha_imaginary, double beta_real,
			double beta_imaginary) {
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Hbmv<cuDoubleComplex> Zhbmv(A_row, A_col, vector_length, sub_diagonals, alpha, beta, 'Z');
  return Zhbmv.HbmvApiCall();
}

int (*cublas_func_ptr[])(int, int, int, int, double, double, double, double) = {
 mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, sub_diagonals, status;
  double alpha_real, alpha_imaginary, beta_real, beta_imaginary;
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

    else if (!(cmd_argument.compare("-sub_diagonals")))
      sub_diagonals = atoi(argv[loop_count + 1]);

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
  if (A_row <= 0 || sub_diagonals <= 0 || sub_diagonals >= A_row) {
    std::cout << "Minimum Dimension error\n";
    return EXIT_FAILURE;
  }

   //! initializing values for A column and vector size
   A_col = A_row;
   vector_length = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, sub_diagonals,
	                               alpha_real, alpha_imaginary, beta_real,
       				               beta_imaginary);

  return status;
}

