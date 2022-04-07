%%writefile ne8.cc
#include <unordered_map>
#include "spmv.h"

template<class T>
Spmv<T>::Spmv(int A_row, int A_col, int vector_length, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Spmv<T>::FreeMemory() {
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
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Spmv<T>::SpmvApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * (A_col + 1)/2];
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

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

  if (!HostVectorY) {
    fprintf (stderr, "!!!! Host memory allocation error (vectorY)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A is a general matrix, X and Y are vectors
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricPackedMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nUpper triangle Matrix A of size " << A_row << " * " << A_col << ":\n";
     util::PrintSymmetricPackedUpperMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeSymmetricPackedMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length);

      std::cout << "\n Upper triangle Matrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, vector_length);
       
      break;
    }

  }
  
  //! Allocating Device Memory for Matrix and Vectors using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceMatrixA, A_row * (A_col + 1)/2 * sizeof(*HostMatrixA));
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
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host matrix to Device matrices using cublasSetMatrix()
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetVector (A_row * (A_col + 1) / 2, sizeof (*HostMatrixA) , (HostMatrixA), VECTOR_LEADING_DIMENSION
                            , DeviceMatrixA, VECTOR_LEADING_DIMENSION);
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

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY,
                           VECTOR_LEADING_DIMENSION, DeviceVectorY, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector Y from host to device failed\n");
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
   * API call to perform the rank-1 update: \f$ A = alpha * X * Y ^ T + A \f$ or  A = alpha * X * Y ^ H + A
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sspmv API\n";
      clk_start = clock();

      status = cublasSspmv(handle,CUBLAS_FILL_MODE_LOWER, A_row, (float *)&alpha, (float *)DeviceMatrixA, (float *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (float *)&beta, (float *)DeviceVectorY, VECTOR_LEADING_DIMENSION);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sspmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sspmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dspmv API\n";
      clk_start = clock();

      status = cublasDspmv(handle,CUBLAS_FILL_MODE_LOWER, A_row, (double *)&alpha, (double *)DeviceMatrixA, (double *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (double *)&beta, (double *)DeviceVectorY, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dspmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dspmv API call ended\n";
      break;
    }

    
  }
  
  //! Copy Vector Y, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY, VECTOR_LEADING_DIMENSION, HostVectorY, 
                           VECTOR_LEADING_DIMENSION);


  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector Y from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "spmv operation is:\n";

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

void mode_S(int A_row, int A_col, int vector_length, double alpha_real, double beta_real) {
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Spmv<float> Sspmv(A_row, A_col, vector_length, alpha, beta, 'S');
  Sspmv.SpmvApiCall();
}

void mode_D(int A_row, int A_col, int vector_length, double alpha_real, double beta_real) {   
  double alpha = alpha_real;
  double beta = beta_real;

  Spmv<double> Dspmv(A_row, A_col, vector_length, alpha, beta, 'D');
  Dspmv.SpmvApiCall();
}




void (*cublas_func_ptr[])(int, int, int, double, double) = {
  mode_S, mode_D
};

int main(int argc, char **argv) {

  int A_row, A_col, vector_length;
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

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! initializing values for matrix B and C
  A_col = A_row;
  vector_length = A_col;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, alpha_real, beta_real);
  
  return 0;
}
