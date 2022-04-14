%%writefile tpr2.cc
#include <unordered_map>
#include "tpr.h"

template<class T>
Tpttr<T>::Tpttr(int A_row, int A_col, char mode) 
    : A_row(A_row), A_col(A_col), mode(mode) {}

template<class T>
void Tpttr<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixAP)
    delete[] HostMatrixAP;
  
  if (HostMatrixA)
    delete[] HostMatrixA;

  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixAP);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for AP" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Tpttr<T>::TpttrApiCall() {
  //! Allocating Host Memory for Packed Matrix 
  int matrix_size = PACKED_MATRIX_SIZE(A_row, A_col);

  HostMatrixAP = new T[matrix_size];
  HostMatrixA = new T[A_row * A_col];

  if (!HostMatrixAP) {
    std::cout << " Host memory allocation error (matrixAP)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostMatrixA) {
    std::cout << " Host memory allocation error (matrixA)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrix based on mode passed,
   * AP is a triangular packed matrix, 
   */
  switch (mode) {
    case 'S': {
      util::InitializeSymmetricPackedMatrix<float>((float *)HostMatrixAP, matrix_size);
      
      std::cout << "\nMatrix AP of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<float>((float *)HostMatrixAP, A_row, matrix_size);           
      break;
    }

    case 'D': {
      util::InitializeSymmetricPackedMatrix<double>((double *)HostMatrixAP, matrix_size);

      std::cout << "\nMatrix AP of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperMatrix<>((double *)HostMatrixAP, A_row, matrix_size);
      break;
    }

    case 'C': {
      util::InitializeSymmetricPackedComplexMatrix<cuComplex>((cuComplex *)HostMatrixAP, matrix_size);
      
      std::cout << "\nMatrix AP of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuComplex >((cuComplex *)HostMatrixAP, A_row, matrix_size);
      break;
    }

    case 'Z': {
      util::InitializeSymmetricPackedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixAP, matrix_size);

      std::cout << "\nMatrix AP of size " << A_row << " * " << A_col << ":\n";
      util::PrintSymmetricPackedUpperComplexMatrix<cuDoubleComplex>((cuDoubleComplex  *)HostMatrixAP, A_row, matrix_size);
       
      break;
    }

  }
  
  //! Allocating Device Memory for Matrix and Vectors using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceMatrixAP, matrix_size * sizeof(*HostMatrixAP));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for AP " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceMatrixA, A_row * A_col * sizeof(*HostMatrixA));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A " << std::endl;
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
  
  //! Copying values of Host matrix to Device matrix using cublasSetVector() and cublasSetMatrix()
  status = cublasSetVector (matrix_size, sizeof (*HostMatrixAP), HostMatrixAP, 
                            VECTOR_LEADING_DIMENSION, DeviceMatrixAP, 
                            VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying matrix AP from host to device  in vector form failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix A from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n <0 \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to perform the conversion from the triangular packed format to the triangular format : \f$ AP -> A \f$ 
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Stpttr API\n";
      clk_start = clock();

      status = cublasStpttr(handle, CUBLAS_FILL_MODE_LOWER, A_row, 
                           (float *)DeviceMatrixAP, (float *)DeviceMatrixA,
                           A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Stpttr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Stpttr API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtpttr API\n";
      clk_start = clock();

      status = cublasDtpttr(handle, CUBLAS_FILL_MODE_LOWER, A_row, 
                           (double *)DeviceMatrixAP, (double *)DeviceMatrixA,
                           A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dtpttr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DTpttr API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctpttr API\n";
      clk_start = clock();

      status = cublasCtpttr(handle, CUBLAS_FILL_MODE_LOWER, A_row, 
                           (cuComplex *)DeviceMatrixAP, (cuComplex *)DeviceMatrixA,
                           A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ctpttr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctpttr API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztpttr API\n";
      clk_start = clock();

      status = cublasZtpttr(handle, CUBLAS_FILL_MODE_LOWER, A_row, 
                           (cuDoubleComplex *)DeviceMatrixAP, (cuDoubleComplex *)DeviceMatrixA,
                           A_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ztpttr kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "ZTpttr API call ended\n";
      break;
    }
  }
  
  
  //! Copy Matrix A, holding resultant Matrix, from Device to Host using cublasGetMatrix()
 status = cublasGetMatrix(A_row, A_col, sizeof(*HostMatrixA),
                           DeviceMatrixA, A_row, HostMatrixA, A_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output matrix A from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix A after " << mode << "Tpttr operation is:\n";

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
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row , A_col);
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row , A_col);
      break;
    }
  }

  long long total_operations = A_row * A_col;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

int mode_S(int A_row, int A_col) {
  Tpttr<float> Stpttr(A_row, A_col, 'S' );

  return Stpttr.TpttrApiCall();
}

int mode_D(int A_row, int A_col) {
  Tpttr<double> Dtpttr(A_row, A_col, 'D');
  return Dtpttr.TpttrApiCall();
}

int mode_C(int A_row, int A_col) {
  Tpttr<cuComplex> Ctpttr(A_row, A_col, 'C');
  return Ctpttr.TpttrApiCall();
}

int mode_Z(int A_row, int A_col) {        
  Tpttr<cuDoubleComplex> Ztpttr(A_row, A_col, 'Z');
  
  return Ztpttr.TpttrApiCall();
}

int (*cublas_func_ptr[])(int, int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, status;
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

  //! Check Dimension Validity
  if (A_row <= 0) {
    std::cout << "Invalid dimension error\n";
    return EXIT_FAILURE;
  }

  //! initializing values for A column 
  A_col = A_row;

  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col);

  return status;
}
