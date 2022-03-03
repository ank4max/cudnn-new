%%writefile ne.cc
#include <unordered_map>
#include "gemv.h"

template<class T>
Gemv<T>::Gemv(int A_row, int A_col, int x_size, int y_size, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), x_size(x_size), y_size(y_size),
      alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Gemv<T>::FreeMemory() {
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
int Gemv<T>::GemvApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[x_size];
  HostVectorY = new T[y_size];

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
   * A is a general matrix, X and Y 
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, x_size);
      util::InitializeVector<float>((float *)HostVectorY, y_size);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintVector<float>((float *)HostVectorX, x_size);
      std::cout << "\nVector Y of size " << y_size << " * 1 : \n" ;
      util::PrintVector<float>((float *)HostVectorY, y_size);
          
      break;
    }

    case 'D': {
      util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, x_size);
      util::InitializeVector<double>((double *)HostVectorY, y_size);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintVector<double>((double *)HostVectorX, x_size);
      std::cout << "\nVector Y of size " << y_size << " * 1 : \n" ;
      util::PrintVector<double>((double *)HostVectorY, y_size);
       
      break;
    }

    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, x_size);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, y_size);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, x_size);
      std::cout << "\nVector Y of size " << y_size << " * 1 : \n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, y_size);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, x_size);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, y_size);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << x_size << " * 1 : \n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, x_size);
      std::cout << "\nVector Y of size " << y_size << " * 1 : \n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, y_size);      
      
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

  cudaStatus = cudaMalloc((void **)&DeviceVectorX, x_size * sizeof(*HostVectorX));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceVectorY, y_size * sizeof(*HostVectorY));
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

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(x_size, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(y_size, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector Y from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * API call to performs matrix - vector multiplication : \f$ Y = alpha * A * X + beta * Y \f$
   */
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - the parameters m, n < 0 or incx, incy = 0
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling SGemv API\n";
      clk_start = clock();

      status = cublasSgemv(handle, CUBLAS_OP_N, A_row, A_col, (float *)&alpha, (float *)DeviceMatrixA, 
                           A_row, (float *)DeviceVectorX, 1, (float *)&beta, (float *)DeviceVectorY, 1);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  SGemv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "SGemv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling DGemv API\n";
      clk_start = clock();

      status = cublasDgemv(handle, CUBLAS_OP_N, A_row, A_col, (double *)&alpha, (double *)DeviceMatrixA, 
                           A_row, (double *)DeviceVectorX, 1, (double *)&beta, (double *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  DGemv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DGemv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling CGemv API\n";
      clk_start = clock();

      status = cublasCgemv(handle, CUBLAS_OP_N, A_row, A_col, (cuComplex  *)&alpha, (cuComplex *)DeviceMatrixA, 
                           A_row, (cuComplex *)DeviceVectorX, 1, (cuComplex *)&beta, (cuComplex *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  CGemv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "CGemv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling ZGemv API\n";
      clk_start = clock();

      status = cublasZgemv(handle, CUBLAS_OP_N, A_row, A_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceMatrixA, 
                           A_row, (cuDoubleComplex *)DeviceVectorX, 1, (cuDoubleComplex *)&beta, (cuDoubleComplex *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  ZGemv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZGemv API call ended\n";
      break;
    }
  }
  
  //! Copy Vector C, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(y_size, sizeof (*HostVectorY), DeviceVectorY, 1, HostVectorY, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector y from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector Y after " << mode << "Gemv operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintVector<float>((float *)HostVectorY, y_size);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorY, y_size);
      break;
    }

    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, y_size);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, y_size);
      break;
    }
  }

  long long total_operations = A_row * A_col * x_size;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(int A_row, int A_col, int x_size, int y_size, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Gemv<float> Sgemv(A_row, A_col, x_size, y_size, alpha, beta, 'S' );
  Sgemv.GemvApiCall();
}

void mode_D(int A_row, int A_col, int x_size, int y_size, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  double alpha = alpha_real;
  double beta = beta_real;

  Gemv<double> Dgemv(A_row, A_col, x_size, y_size, alpha, beta, 'D');
  Dgemv.GemvApiCall();
}

void mode_C(int A_row, int A_col, int x_size, int y_size, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Gemv<cuComplex> CGemv(A_row, A_col, x_size, y_size, alpha, beta, 'C');
  CGemv.GemvApiCall(); 
}

void mode_Z(int A_row, int A_col, int x_size, int y_size, double alpha_real, double alpha_imaginary,
            double beta_real, double beta_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Gemv<cuDoubleComplex> ZGemv(A_row, A_col, x_size, y_size, alpha, beta, 'Z');
  ZGemv.GemvApiCall(); 
}


void (*cublas_func_ptr[])(int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int A_row, A_col, x_size, y_size;
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

    else if (!(cmd_argument.compare("-A_column")))
      A_col = atoi(argv[loop_count + 1]);

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
  
  //! initializing values for vector x and y
  x_size = A_col;
  y_size = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, x_size, y_size, alpha_real, 
                                       alpha_imaginary, beta_real, beta_imaginary);
  
  return 0;
}
