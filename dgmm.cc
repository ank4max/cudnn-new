%%writefile dgmm.cc
#include <unordered_map>
#include "dgmm.h"

template<class T>
Dgmm<T>::Dgmm(int A_row, int A_col, int C_row, int C_col, int vector_length, char mode)
    : A_row(A_row), A_col(A_col), C_row(C_row), C_col(C_col), vector_length(vector_length), mode(mode) {}

template<class T>
void Dgmm<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixA)
    delete[] HostMatrixA;

  if (HostVectorX)
    delete[] HostVectorX;

  if (HostMatrixC)
    delete[] HostMatrixC;


  //! Free Device Memory
  cudaStatus = cudaFree(DeviceMatrixA);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for A" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorX);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for X" << std::endl;
  }

  cudaStatus = cudaFree(DeviceMatrixC);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to uninitialize handle \n";
  }
}

template<class T>
int Dgmm<T>::DgmmApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[vector_length];
  HostMatrixC = new T[C_row * C_col];

  if (!HostMatrixA) {
    std::cout << "!!!! Host memory allocation error (matrixA)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorX) {
    std::cout << "!!!! Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostMatrixC) {
    std::cout << "!!!! Host memory allocation error (matrixC)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  /**
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A and C are general matrices, X is a vector
   */
  switch (mode) {
    case 'S': {
      util::InitializeMatrix<float>((float *)HostMatrixA, A_row, A_col);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeMatrix<float>((float *)HostMatrixC, C_row, C_col);


      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nMatrix C of size " << C_row << " * " << C_col << ":\n";
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
          
      break;
    }

    case 'D': {
      util::InitializeMatrix<double>((double *)HostMatrixA, A_row, A_col);
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeMatrix<double>((double *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintMatrix<double>((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nMatrix A of size " << C_row << " * " << C_col << ":\n";
      util::PrintMatrix<double>((double *)HostMatrixA, C_row, C_col);
       
      break;
    }

    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nMatrix C of size " << C_row << " * " << C_col << ":\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row, C_col);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nMatrix C of size " << C_row << " * " << C_col << ":\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row, C_col);      
      
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

  cudaStatus = cudaMalloc((void **)&DeviceMatrixC, C_row * C_col * sizeof(*HostMatrixC));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  //! Initializing CUBLAS context
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host matrix to Device matrices using cublasSetMatrix()
  //! Copying values of Host vectors to Device vectors using cublasSetVector()
  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix A from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(C_row, C_col, sizeof(*HostMatrixC), HostMatrixC, C_row, DeviceMatrixC, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying matrix C from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  

  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters m,n<0 or mode != CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to performs the matrix-matrix multiplication : \f$ C = A * diag(X) \f$
   */
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sdgmm API\n";
      clk_start = clock();

      status = cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, A_row, A_col,
                           (float *)DeviceMatrixA, A_row, (float *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (float *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Sdgmm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sdgmm API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Ddgmm API\n";
      clk_start = clock();

      status = cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, A_row, A_col,
                           (double *)DeviceMatrixA, A_row, (double *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (double *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Ddgmm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ddgmm API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Cdgmm API\n";
      clk_start = clock();

      status = cublasCdgmm(handle, CUBLAS_SIDE_RIGHT, A_row, A_col,
                           (cuComplex *)DeviceMatrixA, A_row, (cuComplex *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Cdgmm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Cdgmm API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zdgmm API\n";
      clk_start = clock();

      status = cublasZdgmm(handle, CUBLAS_SIDE_RIGHT, A_row, A_col,
                           (cuDoubleComplex *)DeviceMatrixA, A_row, (cuDoubleComplex *)DeviceVectorX,
                           VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceMatrixC, C_row);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "!!!!  Zdgmm kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zdgmm API call ended\n";
      break;
    }
  }
  
  //! Copy Matrix C, holding resultant Matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatrixC),
                           DeviceMatrixC, C_row, HostMatrixC, C_row);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "!!!! Unable to get output matrix C from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix C after " << mode << "dgmm operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintMatrix<float>((float *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'D': {
      util::PrintMatrix<double>((double *)HostMatrixC, C_row, C_col);
      break;
    }

    case 'C': {
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixC, C_row ,C_col);
      break;
    }

    case 'Z': {
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixC, C_row ,C_col);
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

int mode_S(int A_row, int A_col, int C_row, int C_col, int vector_length) {

  Dgmm<float> Sdgmm(A_row, A_col, C_row, C_col, vector_length, 'S');
  return Sdgmm.DgmmApiCall();
}

int mode_D(int A_row, int A_col, int C_row, int C_col, int vector_length) {   

  Dgmm<double> Ddgmm(A_row, A_col, C_row, C_col, vector_length, 'D');
  return Ddgmm.DgmmApiCall();
}

int mode_C(int A_row, int A_col, int C_row, int C_col, int vector_length) {

  Dgmm<cuComplex> Cdgmm(A_row, A_col, C_row, C_col, vector_length, 'C');
  return Cdgmm.DgmmApiCall(); 
}

int mode_Z(int A_row, int A_col, int C_row, int C_col, int vector_length) {

  Dgmm<cuDoubleComplex> Zdgmm(A_row, A_col, C_row, C_col, vector_length, 'Z');
  return Zdgmm.DgmmApiCall(); 
}


int (*cublas_func_ptr[])(int, int, int, int, int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {

  int A_row, A_col, C_row, C_col, vector_length, status;
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

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  //! Dimension check
  if (A_row <= 0 || A_col <= 0) {
    std::cout << "Minimum dimension error\n";
    return EXIT_FAILURE;
  }

  //! initializing values for vector and matrix C
  vector_length = A_col;
  C_row = A_row;
  C_col = A_col;


  status = (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, C_row, C_col, vector_length);
  
  return status;
}
