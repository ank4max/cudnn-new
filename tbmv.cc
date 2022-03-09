%%writefile me.cc
#include <unordered_map>
#include "tbmv.h"

template<class T>
Tbmv<T>::Tbmv(int A_row, int A_col, int vector_length, int super_diagonals,
    int sub_diagonals, char mode) : A_row(A_row), A_col(A_col), 
    vector_length(vector_length), super_diagonals(super_diagonals), 
    sub_diagonals(sub_diagonals), mode(mode) {}

template<class T>
void Tbmv<T>::FreeMemory() {
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
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Tbmv<T>::TbmvApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[vector_length];

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

  /**
   * Switch Case - To Initialize and Print input matrix and vectors based on mode passed,
   * A is a triangular banded matrix, 
   * X is a vector
   */
  switch (mode) {
    case 'S': {
      util::InitializeDiagonalMatrix<float>((float *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintDiagonalMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeDiagonalMatrix<double>((double *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeVector<double >((double *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintDiagonalMatrix<double >((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double >((double  *)HostVectorX, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeComplexDiagonalMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexDiagonalMatrix<cuComplex >((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);  
       
      break;
    }

    case 'Z': {
      util::InitializeComplexDiagonalMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintComplexDiagonalMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
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
    std::cout << " Failed to initialize handle\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  //! Copying values of Host matrix to Device matrix using cublasSetMatrix()
  //! Copying values of Host vectors to Device vectors using cublasSetVector()

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying matrix A from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, 
                           VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_INVALID_VALUE - The parameters n, k<0 or incx=0 \n
   * CUBLAS_STATUS_ALLOC_FAILED - The allocation of internal scratch memory failed \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to perform the triangular banded matrix-vector multiplication : \f$ X = op(A) * X \f$ 
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Stbmv API\n";
      clk_start = clock();

      status = cublasStbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (float *)DeviceMatrixA, A_row, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Stbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Stbmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtbmv API\n";
      clk_start = clock();

      status = cublasDtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (double *)DeviceMatrixA, A_row, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dtbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtbmv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctbmv API\n";
      clk_start = clock();

      status = cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (cuComplex *)DeviceMatrixA, A_row, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ctbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctbmv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztbmv API\n";
      clk_start = clock();

      status = cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (cuDoubleComplex *)DeviceMatrixA, A_row, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ztbmv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztbmv API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X, holding resultant vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector X from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nVector X after " << mode << "tbmv operation is:\n";

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


void mode_S(int A_row, int A_col, int vector_length, int super_diagonals, 
            int sub_diagonals) {

  Tbmv<float> Stbmv(A_row, A_col, vector_length, super_diagonals, sub_diagonals, 'S' );
  Stbmv.TbmvApiCall();
}

void mode_D(int A_row, int A_col, int vector_length, int super_diagonals, 
            int sub_diagonals) {            

  Tbmv<double> Dtbmv(A_row, A_col, vector_length, super_diagonals, sub_diagonals, 'D');
  Dtbmv.TbmvApiCall();
}

void mode_C(int A_row, int A_col, int vector_length, int super_diagonals, 
            int sub_diagonals) {
            
  Tbmv<cuComplex> Ctbmv(A_row, A_col, vector_length, super_diagonals, sub_diagonals, 'C');
  Ctbmv.TbmvApiCall();
}

void mode_Z(int A_row, int A_col, int vector_length, int super_diagonals, 
            int sub_diagonals) {
            
  Tbmv<cuDoubleComplex> Ztbmv(A_row, A_col, vector_length, super_diagonals, sub_diagonals, 'Z');
  Ztbmv.TbmvApiCall();
}

void (*cublas_func_ptr[])(int, int, int, int, int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, sub_diagonals, super_diagonals;
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

    else if (!(cmd_argument.compare("-sub_diagonals")))
      sub_diagonals = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

   //! initializing values for A column, super_diagonals and vector size
  A_col = A_row;
  vector_length = A_row;

  //! For triangular banded matrix stored in lower mode
  super_diagonals = 0;  

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, super_diagonals,
                                       sub_diagonals);

  return 0;
}
