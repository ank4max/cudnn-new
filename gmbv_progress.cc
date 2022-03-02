%%writefile na.cc
#include <unordered_map>
#include "gbmv.h"

template<class T>
Gbmv<T>::Gbmv(int A_row, int A_col, int x_size, int y_size,  int super_diagonals, int sub_diagonals, T alpha, T beta, char mode)
    : A_row(A_row), A_col(A_col), x_size(x_size), y_size(y_size),
      super_diagonals(super_diagonals), sub_diagonals(sub_diagonals), alpha(alpha), beta(beta), mode(mode) {}

template<class T>
void Gbmv<T>::FreeMemory() {
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
    std::cout << " The device memory deallocation failed for B" << std::endl;
  }

  cudaStatus = cudaFree(DeviceVectorY);
  if (cudaStatus != cudaSuccess) {
    std::cout << " The device memory deallocation failed for C" << std::endl;
  }

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
  }
}

template<class T>
int Gbmv<T>::GbmvApiCall() {
  //! Allocating Host Memory for Matrices
  HostMatrixA = new T[A_row * A_col];
  HostVectorX = new T[x_size];
  HostVectorY = new T[y_size];

  if (!HostMatrixA) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixA)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  if (!HostVectorX) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixB)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    fprintf (stderr, "!!!! Host memory allocation error (matrixC)\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input matrices based on mode passed,
   * A, B and C are general matrices
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeDiagonalMatrix<float>((float *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeVector<float>((float *)HostVectorX, x_size);
      util::InitializeVector<float>((float *)HostVectorY, y_size);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<float>((float *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X:\n";
      util::PrintVector<float>((float *)HostVectorX, x_size);
      std::cout << "\nVector Y:\n";
      util::PrintVector<float>((float *)HostVectorY, y_size);
          
      break;
    }

    case 'D': {
      util::InitializeDiagonalMatrix<double>((double *)HostMatrixA, A_row, A_col, super_diagonals, sub_diagonals);
      util::InitializeVector<double >((double *)HostVectorX, x_size);
      util::InitializeVector<double >((double  *)HostVectorY, y_size);

      std::cout << "\nMatrix A:\n";
      util::PrintMatrix<double >((double *)HostMatrixA, A_row, A_col);
      std::cout << "\nVector X:\n";
      util::PrintVector<double >((double  *)HostVectorX, x_size);
      std::cout << "\nVector Y:\n";
      util::PrintVector<double >((double  *)HostVectorY, y_size);
       
      break;
    }
/*
    case 'C': {
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostVectorX, B_row, B_col);
      util::InitializeComplexMatrix<cuComplex>((cuComplex *)HostVectorY, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostVectorX, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuComplex>((cuComplex *)HostVectorY, C_row, C_col);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, B_row, B_col);
      util::InitializeComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, C_row, C_col);

      std::cout << "\nMatrix A:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostMatrixA, A_row, A_col);
      std::cout << "\nMatrix B:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, B_row, B_col);
      std::cout << "\nMatrix C:\n";
      util::PrintComplexMatrix<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, C_row, C_col);      
      
      break;
    }

    */
  }
  
  //! Allocating Device Memory for Matrices using cudaMalloc()
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
  
  //! Copying values of Host matrices to Device matrices using cublasSetMatrix()

  status = cublasSetMatrix(A_row, A_col, sizeof(*HostMatrixA), HostMatrixA, A_row, DeviceMatrixA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(x_size, sizeof(*HostVectorX), HostVectorX, 1, DeviceVectorX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector x from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(y_size, sizeof(*HostVectorY), HostVectorY, 1, DeviceVectorY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying vector y from host to device failed\n");
    FreeMemory();
    return EXIT_FAILURE;
  }
  
  
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Sgbmv API\n";
      clk_start = clock();

      status = cublasSgbmv(handle, CUBLAS_OP_N, A_row, A_col, super_diagonals, sub_diagonals, (float *)&alpha, 
                           (float *)DeviceMatrixA, A_row, (float *)DeviceVectorX, 1, (float *)&beta, (float *)DeviceVectorY, 1);


      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  Sgbmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Sgbmv API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Dgbmv API\n";
      clk_start = clock();

      status = cublasDgbmv(handle, CUBLAS_OP_N, A_row, A_col, super_diagonals, sub_diagonals, (double *)&alpha, 
                           (double *)DeviceMatrixA, A_row, (double *)DeviceVectorX, 1, (double *)&beta, (double *)DeviceVectorY, 1);

      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!!  DGbmv kernel execution error\n");
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "DGbmv API call ended\n";
      break;
    }

    
  }
  
  //! Copy Matrix C, holding resultant matrix, from Device to Host using cublasGetMatrix()
  status = cublasGetVector(y_size, sizeof (*HostVectorY), DeviceVectorY, 1, HostVectorY, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to get output matrix C from device\n");
    FreeMemory();
    return EXIT_FAILURE;
  }

  std::cout << "\nMatrix C after " << mode << "Gbmv operation is:\n";

  switch (mode) {
    case 'S': {  
      util::PrintVector<float>((float *)HostVectorY, y_size);
      break;
    }

    case 'D': {
      util::PrintVector<double>((double *)HostVectorY, y_size);
      break;
    }
/*
    case 'C': {
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, y_size);
      break;
    }

    case 'Z': {
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, y_size);
      break;
    }
    */
  }

  long long total_operations = A_row * A_col * x_size;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}


void mode_S(int A_row, int A_col, int x_size, int y_size, int super_diagonals, int sub_diagonals, 
            double alpha_real, double alpha_imaginary, double beta_real, double beta_imaginary) {
            
  float alpha = (float)alpha_real;
  float beta = (float)beta_real;

  Gbmv<float> Sgbmv(A_row, A_col, x_size, y_size, super_diagonals, sub_diagonals, alpha, beta, 'S' );
  Sgbmv.GbmvApiCall();
}

void mode_D(int A_row, int A_col, int x_size, int y_size, int super_diagonals, int sub_diagonals, 
            double alpha_real, double alpha_imaginary, double beta_real, double beta_imaginary) {
            
  double alpha = alpha_real;
  double beta = beta_real;

  Gbmv<double> DGbmv(A_row, A_col, x_size, y_size, super_diagonals, sub_diagonals, alpha, beta, 'D');
  DGbmv.GbmvApiCall();
}

void mode_C(int A_row, int A_col, int x_size, int y_size, int super_diagonals, int sub_diagonals, 
            double alpha_real, double alpha_imaginary, double beta_real, double beta_imaginary) {
            
  cuComplex alpha = {(float)alpha_real, (float)alpha_imaginary};
  cuComplex beta = {(float)beta_real, (float)beta_imaginary};

  Gbmv<cuComplex> CGbmv(A_row, A_col, x_size, y_size, super_diagonals, sub_diagonals, alpha, beta, 'C');
  CGbmv.GbmvApiCall(); 
}

void mode_Z(int A_row, int A_col, int x_size, int y_size, int super_diagonals, int sub_diagonals, 
            double alpha_real, double alpha_imaginary, double beta_real, double beta_imaginary) {
            
  cuDoubleComplex alpha = {alpha_real, alpha_imaginary};
  cuDoubleComplex beta = {beta_real, beta_imaginary};

  Gbmv<cuDoubleComplex> ZGbmv(A_row, A_col, x_size, y_size, super_diagonals, sub_diagonals, alpha, beta, 'Z');
  ZGbmv.GbmvApiCall(); 
}


void (*cublas_func_ptr[])(int, int, int, int, int, int, double, double, double, double) = {
  mode_S, mode_D, mode_C, mode_Z
};







int main(int argc, char **argv) {

  int A_row, A_col, x_size, y_size, sub_diagonals, super_diagonals;
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

    else if (!(cmd_argument.compare("-super_diagonals")))
      super_diagonals = atoi(argv[loop_count + 1]);

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

   //! initializing values for matrix B and C
  x_size = A_col;
  y_size = A_row;

  (*cublas_func_ptr[mode_index[mode]])(A_row, A_col, x_size, y_size, super_diagonals, sub_diagonals, alpha_real, alpha_imaginary, beta_real, beta_imaginary);
 

  
 

  
  return 0;
}

