%%writefile cublas_sbmv_test2.cc
#include <unordered_map>
#include "cublas_sbmv_test.h"

template<class T>
Sbmv<T>::Sbmv(int A_row, int A_col, int vector_length, int sub_diagonals,
	T alpha, T beta, char mode) : A_row(A_row), A_col(A_col),
	vector_length(vector_length), sub_diagonals(sub_diagonals),
	alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Sbmv<T>::FreeMemory() {
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
int Sbmv<T>::SbmvApiCall() {
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
   * A is a symmetric banded matrix,
   * X and Y are vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeTriangularBandedMatrix<float>((float *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedMatrix<float>((float *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);

      break;
    }

    case 'D': {
      util::InitializeTriangularBandedMatrix<double>((double *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeVector<double >((double *)HostVectorX, vector_length);
      util::InitializeVector<double >((double  *)HostVectorY, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedMatrix<double>((double *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double >((double  *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double >((double  *)HostVectorY, vector_length);

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
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - the parameters m, n < 0 or incx, incy = 0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */

  /**
   * API call to perform the symmetric banded matrix-vector multiplication: \f$ Y = alpha * A * X + beta * Y \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Ssbmv API\n";
      clk_start = clock();

      status = cublasSsbmv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
		           sub_diagonals, (float *)&alpha, (float *)DeviceMatrixA,
         		   A_row, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (float *)&beta, (float *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ssbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ssbmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dsbmv API\n";
      clk_start = clock();

      status = cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, vector_length,
                           sub_diagonals, (double *)&alpha, (double *)DeviceMatrixA,
			   A_row, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
			   (double *)&beta, (double *)DeviceVectorY,
			   VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dsbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dsbmv API call ended\n";
      break;
    }
  }

  //! Copy Vector Y, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY,
                           VECTOR_LEADING_DIMENSION, HostVectorY,
			   VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector Y from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "sbmv operation is:\n";
  switch (mode) {
    case 'S': {
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorY, vector_length);
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


int mode_S(int A_row, int A_col, int vector_length, int sub_diagonals, double alpha_real, double beta_real) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Sbmv<float> Ssbmv(A_row, A_col, vector_length, sub_diagonals, alpha, beta, 'S' );
  return Ssbmv.SbmvApiCall();
}

int mode_D(int A_row, int A_col, int vector_length, int sub_diagonals, double alpha_real, double beta_real) {
  double alpha = alpha_real;
  double beta = beta_real;

  Sbmv<double> Dsbmv(A_row, A_col, vector_length, sub_diagonals, alpha, beta, 'D');
  return Dsbmv.SbmvApiCall();
}

int (*cublas_func_ptr[])(int, int, int, int, double, double) = {
  mode_S, mode_D
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, sub_diagonals, status;
  double alpha_real, beta_real;
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

    else if (!(cmd_argument.compare("-sub_diagonals")))
      sub_diagonals = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

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

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, sub_diagonals, alpha_real, beta_real);
  return status;
}
