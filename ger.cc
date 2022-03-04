%%writefile ne.cc
#include <unordered_map>
#include "ger.h"

template<class T>
Ger<T>::Ger(int A_row, int A_col, int X_length, int Y_length, T alpha, char mode)
    : A_row(A_row), A_col(A_col), X_length(X_length), Y_length(Y_length),
      alpha(alpha), mode(mode) {}

template<class T>
void Ger<T>::FreeMemory() {
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
int Ger<T>::GerApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[X_length];
  HostVectorY = new T[Y_length];

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
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, X_length);
      util::InitializeVector<float>((float *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, Y_length);
          
      break;
    }

    case 'D': {
      util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, X_length);
      util::InitializeVector<double>((double *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, Y_length);
       
      break;
    }

    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, X_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, Y_length);
      
      break;
    }

    case 'H': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, X_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, Y_length);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, X_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, Y_length);      
      
      break;
    }

    case 'T': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, X_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, Y_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << X_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, X_length);
      std::cout << "\nVector Y of size " << Y_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, Y_length);      
      
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

  cudaStatus = cudaMalloc((void **)&DeviceVectorX, X_length * sizeof(*HostVectorX));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceVectorY, Y_length * sizeof(*HostVectorY));
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

  status = cublasSetVector(X_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector X from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(Y_length, sizeof(*HostVectorY), HostVectorY,
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
      std::cout << "\nCalling Sger API\n";
      clk_start = clock();

      status = cublasSger(handle, A_row, A_col, (float *)&alpha, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (float *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (float *)DeviceMatrixA, A_row);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sger kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sger API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dger API\n";
      clk_start = clock();

      status = cublasDger(handle, A_row, A_col, (double *)&alpha, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (double *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (double *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  DGer kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DGer API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cgeru API\n";
      clk_start = clock();

      status = cublasCgeru(handle, A_row, A_col, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cgeru kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgeru API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Cgerc API\n";
      clk_start = clock();

      status = cublasCgerc(handle, A_row, A_col, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Cgerc kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cgerc API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zgeru API\n";
      clk_start = clock();

      status = cublasZgeru(handle, A_row, A_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zgeru kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zgeru API call ended\n";
      break;
    }

    case 'T': {
      std::cout << "\nCalling Zgerc API\n";
      clk_start = clock();

      status = cublasZgerc(handle, A_row, A_col, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuDoubleComplex *)DeviceVectorY, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zgerc kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zgerc API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix A, holding resultant Matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(A_row, A_col, sizeof (*HostMatrixA), DeviceMatrixA, 
                           A_row, HostMatrixA, A_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output vector Y from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "ger operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'D': {
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'H': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'T': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      break;
    }
  }

  long long total_operations = A_row * X_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(int A_row, int A_col, int X_length, int Y_length, double alpha_real, double alpha_imaginary) {
  float alpha = (float)alpha_real;

  Ger<float> Sger(A_row, A_col, X_length, Y_length, alpha, 'S');
  Sger.GerApiCall();
}

void mode_D(int A_row, int A_col, int X_length, int Y_length, double alpha_real, 
            double alpha_imaginary) {   
  double alpha = alpha_real;

  Ger<double> Dger(A_row, A_col, X_length, Y_length, alpha, 'D');
  Dger.GerApiCall();
}

void mode_C(int A_row, int A_col, int X_length, int Y_length, double alpha_real,
            double alpha_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Ger<cuComplex> Cgeru(A_row, A_col, X_length, Y_length, alpha, 'C');
  Cgeru.GerApiCall(); 
}

void mode_H(int A_row, int A_col, int X_length, int Y_length, double alpha_real,
            double alpha_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Ger<cuComplex> Cgerc(A_row, A_col, X_length, Y_length, alpha, 'H');
  Cgerc.GerApiCall(); 
}

void mode_Z(int A_row, int A_col, int X_length, int Y_length, double alpha_real,
            double alpha_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  Ger<cuDoubleComplex> Zgeru(A_row, A_col, X_length, Y_length, alpha, 'Z');
  Zgeru.GerApiCall(); 
}

void mode_T(int A_row, int A_col, int X_length, int Y_length, double alpha_real,
            double alpha_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  Ger<cuDoubleComplex> Zgerc(A_row, A_col, X_length, Y_length, alpha, 'T');
  Zgerc.GerApiCall(); 
}


void (*cublas_func_ptr[])(int, int, int, int, double, double) = {
  mode_S, mode_D, mode_C, mode_H, mode_Z, mode_T
};

int main(int argc, char **argv) {

  int A_row, A_col, X_length, Y_length;
  double alpha_real, alpha_imaginary;
  char mode;
    
  std::unordered_map<char, int> mode_index;
  mode_index['S'] = 0;
  mode_index['D'] = 1;
  mode_index['C'] = 2;
  mode_index['H'] = 3;
  mode_index['Z'] = 4;
  mode_index['T'] = 5;

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

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! initializing values for matrix B and C
  X_length = A_col;
  Y_length = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, X_length, Y_length, alpha_real, 
                                       alpha_imaginary);
  
  return 0;
}
