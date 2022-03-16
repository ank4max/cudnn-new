%%writefile test2.cc
#include <unordered_map>
#include "rotmg.h"

template<class T>
Rotmg<T>::Rotmg(T scalar_d1, T scalar_d2, T scalar_x1, T scalar_y1, char mode) : scalar_d1(scalar_d1),
              scalar_d2(scalar_d2), scalar_x1(scalar_x1), scalar_y1(scalar_y1), mode(mode) {}

template<class T>
void Rotmg<T>::FreeMemory() {
  //! Free Host Memory
  if (HostMatrixParam)
    delete[] HostMatrixParam;

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Rotmg<T>::RotmgApiCall() {
  //! Allocating Host Memory for Matrix and Vectors
  int matrix_size = 5;
  HostMatrixParam = new T[matrix_size];

  if (!HostMatrixParam) {
    std::cout << " Host memory allocation error (matrixParam)\n";
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
    
  /**
   * The Error values returned by API are : \n
   * CUBLAS_STATUS_SUCCESS - The operation completed successfully \n
   * CUBLAS_STATUS_NOT_INITIALIZED - The library was not initialized \n
   * CUBLAS_STATUS_EXECUTION_FAILED - The function failed to launch on the GPU \n
   */
    
  /**
   * API call to constructs the modified Givens transformation :
   * The result is that zeros out the second entry of a 2×1 vector(d1 * 1/2 * x1, d2 * 1/2 * y1) * T.
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
   * The elements , , and of matrix H are stored in param[1], param[2], param[3] and param[4], respectively. 
   */
  switch (mode) {

    case 'S': {
      std::cout << "\nCalling Srotmg API\n";
      std::cout <<"\nx1 = " << scalar_x1;
      clk_start = clock();

      status = cublasSrotmg(handle, (float *)&scalar_d1, (float *)&scalar_d2, (float *)&scalar_x1, (float *)&scalar_y1, (float *)HostMatrixParam);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Srotmg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Srotmg API call ended\n";
      std::cout <<"\nx1 = "<<scalar_x1;
      break;
    }

    case 'D': {
      std::cout << "\nCalling Drotmg API\n";
      clk_start = clock();

      status = cublasDrotmg(handle, (double *)&scalar_d1, (double *)&scalar_d2, (double *)&scalar_x1, (double *)&scalar_y1, (double *)HostMatrixParam);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Drotmg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Drotmg API call ended\n";
      break;
    }
  }
  

  switch (mode) {

    case 'S': {
      std::cout << "\nHostMatrixParam[0] after " << mode << "rotmg operation is: ";
      std::cout << HostMatrixParam[0];

      std::cout << "\nh11 : " << HostMatrixParam [1];
      std::cout << "\nh22 : " << HostMatrixParam [4];
      std::cout <<"x1 = "<< scalar_x1;

      std::cout << "\nChecking the second entry of H*{ sqrt {d1 )*x1 , sqrt {d2 }* y1 }^T : " ;
      float result ;
      result =  ( -1.0)* sqrt(scalar_d1) * 1 + HostMatrixParam[4] * sqrt(scalar_d2) * (scalar_y1);
      std::cout << result << std::endl;
      break;
    }

    case 'D': {
      std::cout << "\nHostMatrixParam[0] after " << mode << "rotmg operation is: ";
      std::cout << HostMatrixParam[0];

      std::cout << "\nh11 : " << HostMatrixParam [1];
      std::cout << "\nh22 : " << HostMatrixParam [4];
      std::cout << "\nChecking the second entry of H*{sqrt {d1 )*x1 , sqrt {d2 }* y1 }^T : "  ;
      double result ;
      result =  ( -1.0) * sqrt(scalar_d1) * 1 + HostMatrixParam[4] * sqrt(scalar_d2) * (scalar_y1);
      std::cout << result << std::endl;
      break;
    }
  }



  long long total_operations = (matrix_size - 1) * 2 ;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}


void mode_S(double scalar_d1, double scalar_d2,
            double scalar_x1, double scalar_y1) {
            
  float param_d1 = (float)scalar_d1;
  float param_d2 = (float)scalar_d2;
  float param_x1 = (float)scalar_x1;
  float param_y1 = (float)scalar_y1;

  Rotmg<float> Srotmg(param_d1, param_d2, param_x1, param_y1, 'S');
  Srotmg.RotmgApiCall(); 
}

void mode_D(double scalar_d1, double scalar_d2,
            double scalar_x1, double scalar_y1) {
            

  Rotmg<double> Drotmg(scalar_d1, scalar_d2, scalar_x1, scalar_y1, 'D');
  Drotmg.RotmgApiCall(); 
}

void (*cublas_func_ptr[])(double, double, double, double) = {
 mode_S, mode_D
};

int main(int argc, char **argv) {
  double scalar_d1, scalar_d2, scalar_x1, scalar_y1;
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

    if (!(cmd_argument.compare("-scalar_d1")))
      scalar_d1 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_d2")))
      scalar_d2 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_x1")))
      scalar_x1 = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_y1"))) 
      scalar_y1 = std::stod(argv[loop_count + 1]);   
  
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  (*cublas_func_ptr[mode_index[mode]])(scalar_d1, scalar_d2, 
                                       scalar_x1, scalar_y1);

  return 0;
}
