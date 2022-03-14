%%writefile max.cc
#include <unordered_map>
#include "tbsv.h"

template<class T>
Tbsv<T>::Tbsv(int A_row, int A_col, int vector_length, int sub_diagonals, char mode) : A_row(A_row), A_col(A_col), 
    vector_length(vector_length), sub_diagonals(sub_diagonals), mode(mode) {}

template<class T>
void Tbsv<T>::FreeMemory() {
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
int Tbsv<T>::TbsvApiCall() {
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
      util::InitializeTriangularBandedMatrix<float>((float *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedMatrix<float>((float *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeTriangularBandedMatrix<double>((double *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeVector<double >((double *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedMatrix<double>((double *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double >((double  *)HostVectorX, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeTriangularBandedComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeComplexVector<cuComplex >((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedComplexMatrix<cuComplex >((cuComplex *)HostMatrixA, A_row);
      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);  
       
      break;
    }

    case 'Z': {
      util::InitializeTriangularBandedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, sub_diagonals);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nMatrix A of size " << A_row << " * " << A_col << ":\n";
      util::PrintTriangularBandedComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row);
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
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
  
  /**
   * API call to solve the triangular banded linear system with a single right-hand-side : \f$ op(A) * X = b \f$ 
   * The solution x overwrites the right-hand-sides b on exit.
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Stbsv API\n";
      clk_start = clock();

      status = cublasStbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (float *)DeviceMatrixA, A_row, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Stbsv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Stbsv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dtbsv API\n";
      clk_start = clock();

      status = cublasDtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (double *)DeviceMatrixA, A_row, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Dtbsv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Dtbsv API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Ctbsv API\n";
      clk_start = clock();

      status = cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (cuComplex *)DeviceMatrixA, A_row, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ctbsv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ctbsv API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Ztbsv API\n";
      clk_start = clock();

      status = cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, A_row, sub_diagonals,
                           (cuDoubleComplex *)DeviceMatrixA, A_row, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Ztbsv kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Ztbsv API call ended\n";
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

  std::cout << "\nVector X after " << mode << "tbsv operation is:\n";

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


void mode_S(int A_row, int A_col, int vector_length, int sub_diagonals) {

  Tbsv<float> Stbsv(A_row, A_col, vector_length, sub_diagonals, 'S' );
  Stbsv.TbsvApiCall();
}

void mode_D(int A_row, int A_col, int vector_length, int sub_diagonals) {            

  Tbsv<double> Dtbsv(A_row, A_col, vector_length, sub_diagonals, 'D');
  Dtbsv.TbsvApiCall();
}

void mode_C(int A_row, int A_col, int vector_length, int sub_diagonals) {
            
  Tbsv<cuComplex> Ctbsv(A_row, A_col, vector_length, sub_diagonals, 'C');
  Ctbsv.TbsvApiCall();
}

void mode_Z(int A_row, int A_col, int vector_length, int sub_diagonals) {
            
  Tbsv<cuDoubleComplex> Ztbsv(A_row, A_col, vector_length, sub_diagonals, 'Z');
  Ztbsv.TbsvApiCall();
}

void (*cublas_func_ptr[])(int, int, int, int) = {
  mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  int A_row, A_col, vector_length, sub_diagonals;
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

   //! initializing values for A column, and vector size
  A_col = A_row;
  vector_length = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, vector_length, sub_diagonals);

  return 0;
}
