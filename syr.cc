%%writefile max.cc
#include <unordered_map>
#include "syr.h"

template<class T>
Syr<T>::Syr(int A_row, int A_col, int vector_length, T alpha, char mode)
    : A_row(A_row), A_col(A_col), vector_length(vector_length), 
      alpha(alpha), mode(mode) {}

template<class T>
void Syr<T>::FreeMemory() {
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
int Syr<T>::SyrApiCall() {
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
   * A is a symmetric matrix, X is a vector
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
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to performs the symmetric rank-1 update : \f$ A = alpha * X * X ^ T + A \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Ssyr API\n";
      clk_start = clock();

      status = cublasSsyr(handle, CUBLAS_FILL_MODE_LOWER, vector_length, (float *)&alpha, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (float *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Ssyr kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ssyr API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dsyr API\n";
      clk_start = clock();

      status = cublasDsyr(handle, CUBLAS_FILL_MODE_LOWER, vector_length, (double *)&alpha, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (double *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Dsyr kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dsyr API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Csyr API\n";
      clk_start = clock();

      status = cublasCsyr(handle, CUBLAS_FILL_MODE_LOWER, vector_length, (cuComplex *)&alpha, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Csyr kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Csyr API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zsyr API\n";
      clk_start = clock();

      status = cublasZsyr(handle, CUBLAS_FILL_MODE_LOWER, vector_length, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION,
                          (cuDoubleComplex *)DeviceMatrixA, A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Zsyr kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zsyr API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix A, holding resultant Matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(A_row, A_col, sizeof(*HostMatrixA),
                           DeviceMatrixA, A_row, HostMatrixA, A_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix A from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "syr operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintSymmetricMatrix<float>((float *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'D': {
      util::PrintSymmetricMatrix<double>((double *)HostMatrixA, A_row, A_col);
      break;
    }

    case 'C': {
      util::PrintSymmetricComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row ,A_col); 
      break;
    }

    case 'Z': {
      util::PrintSymmetricComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row ,A_col);
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

void mode_S(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {
  float alpha = (float)alpha_real;

  Syr<float> Ssyr(A_row, A_col, vector_length, alpha, 'S');
  Ssyr.SyrApiCall();
}

void mode_D(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {   
  double alpha = alpha_real;

  Syr<double> Dsyr(A_row, A_col, vector_length, alpha, 'D');
  Dsyr.SyrApiCall();
}

void mode_C(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};

  Syr<cuComplex> Csyr(A_row, A_col, vector_length, alpha, 'C');
  Csyr.SyrApiCall(); 
}

void mode_Z(int A_row, int A_col, int vector_length, double alpha_real, double alpha_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};

  Syr<cuDoubleComplex> Zsyr(A_row, A_col, vector_length, alpha, 'Z');
  Zsyr.SyrApiCall(); 
}


void (*cublas_func_ptr[])(int, int, int, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int A_row, A_col, vector_length;
  double alpha_real, alpha_imaginary;
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

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! initializing values for A column and vector size
  A_col = A_row;
  vector_length = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, alpha_real, 
                                       alpha_imaginary);
  
  return 0;
}
