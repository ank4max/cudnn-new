%%writefile max9.cc
#include <unordered_map>
#include "rotg.h"

template<class T>
Rotg<T>::Rotg(T scalar_a, T scalar_b, char mode) : scalar_a(scalar_a),
              scalar_b(scalar_b), mode(mode) {}

template<class T>
void Rotg<T>::FreeMemory() {

  //! Destroy CuBLAS context
  status  = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to uninitialize handle" << std::endl;
  }
}

template<class T>
int Rotg<T>::RotgApiCall() {

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
      std::cout << "\nCalling Srotg API\n";
      //float sine = (float) sine;
      clk_start = clock();

      status = cublasSrotg(handle, (float *)&scalar_a, (float *)&scalar_b, (float *)&cosine, (float *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Srotg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Srotg API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Drotg API\n";
      double sine = (double)sine;
      clk_start = clock();

      status = cublasDrotg(handle, (double *)&scalar_a, (double *)&scalar_b, (double *)&cosine, (double *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Drotg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Drotg API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Crotg API\n";
      clk_start = clock();

      status = cublasCrotg(handle, (cuComplex *)&scalar_a, (cuComplex *)&scalar_b, (float *)&cosine, (cuComplex *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Crotg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Crotg API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zrotg API\n";
      clk_start = clock();

      status = cublasZrotg(handle, (cuDoubleComplex *)&scalar_a, (cuDoubleComplex *)&scalar_b, (double *)&cosine, (cuDoubleComplex *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zrotg kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zrotg API call ended\n";
      break;
      
    }
    
  }

  switch(mode) {      
    case 'S' : {
      util::PrintRotgOutput<float, float>((float *)&scalar_a, (float *)&sine, (float*)&cosine, mode); 
      break;
    }

    case 'D' : {
      util::PrintRotgOutput<double, double>((double *)&scalar_a, (double *)&sine, (double *)&cosine, mode); 
      break;
    }

    case 'C' : {
      util::PrintRotgComplexOutput<cuComplex, float>((cuComplex *)&scalar_a, (cuComplex *)&sine, (float *)&cosine, mode);
      break;  
      }

    case 'Z' : {
      util::PrintRotgComplexOutput<cuDoubleComplex, double>((cuDoubleComplex *)&scalar_a, (cuDoubleComplex *)&sine, (double *)&cosine, mode);
      break;
    }
  }

  long long total_operations = (4 * 2) ;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}


void mode_S(double scalar_a_real, double scalar_a_imaginary, double scalar_b_real, double scalar_b_imaginary) {
            
  float scalar_a = (float)scalar_a_real;
  float scalar_b = (float)scalar_b_real;
  
  Rotg<float> Srotg(scalar_a, scalar_b, 'S');
  Srotg.RotgApiCall(); 
}

void mode_D(double scalar_a_real, double scalar_a_imaginary, double scalar_b_real, double scalar_b_imaginary) {
           
  Rotg<double> Drotg(scalar_a_real, scalar_b_real, 'D');
  Drotg.RotgApiCall(); 
}

void mode_C(double scalar_a_real, double scalar_a_imaginary, double scalar_b_real, double scalar_b_imaginary) {
            
  cuComplex scalar_a = {(float) scalar_a_real, (float)scalar_a_imaginary};
  cuComplex scalar_b = {(float)scalar_b_real, (float)scalar_b_imaginary};

  Rotg<cuComplex> Crotg(scalar_a, scalar_b, 'C');
  Crotg.RotgApiCall(); 
}

void mode_Z(double scalar_a_real, double scalar_a_imaginary, double scalar_b_real, double scalar_b_imaginary) {
            
  cuDoubleComplex scalar_a = {scalar_a_real, scalar_a_imaginary};
  cuDoubleComplex scalar_b = {scalar_b_real, scalar_b_imaginary};

  Rotg<cuDoubleComplex> Zrotg(scalar_a, scalar_b, 'Z');
  Zrotg.RotgApiCall(); 
} 

void (*cublas_func_ptr[])(double, double, double, double) = {
 mode_S, mode_D, mode_C, mode_Z
};

int main(int argc, char **argv) {
  double scalar_a_real, scalar_a_imaginary, scalar_b_real, scalar_b_imaginary;
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

    if (!(cmd_argument.compare("-scalar_a_real")))
      scalar_a_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_a_imaginary")))
      scalar_a_imaginary = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_b_real")))
      scalar_b_real = std::stod(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-scalar_b_imaginary")))
      scalar_b_imaginary = std::stod(argv[loop_count + 1]);  
  
    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }

  (*cublas_func_ptr[mode_index[mode]])(scalar_a_real, scalar_a_imaginary, scalar_b_real, scalar_b_imaginary);

  return 0;
}
