%%writefile max6.cc
#include <unordered_map>
#include "rot.h"

template<class T, class C, class D>
Rot<T, C, D>::Rot(int vector_length, C sine, D cosine, char mode)
              : vector_length(vector_length), sine(sine), cosine(cosine), mode(mode) {}

template<class T, class C, class D>
void Rot<T, C, D>::FreeMemory() {
  //! Free Host Memory
  if (HostVectorX)
    delete[] HostVectorX;
  
  if (HostVectorY)
    delete[] HostVectorY;

  //! Free Device Memory
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
    std::cout << " Unable to uninitialize handle \n";
  }
}

template<class T, class C, class D>
int Rot<T, C, D>::RotApiCall() {
  //! Allocating Host Memory for Vectors
  HostVectorX = new T[vector_length];
  HostVectorY = new T[vector_length];

  if (!HostVectorX) {
    std::cout << " Host memory allocation error (vectorX)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  if (!HostVectorY) {
    std::cout << " Host memory allocation error (vectorY)\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  /**
   * Switch Case - To Initialize and Print input vectors based on mode passed,
   * X and Y are vectors
   */
  
  switch (mode) {
    case 'S': {
      util::InitializeVector<float>((float *)HostVectorX, vector_length);
      util::InitializeVector<float>((float *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<float>((float *)HostVectorY, vector_length);
          
      break;
    }

    case 'D': {
      util::InitializeVector<double>((double *)HostVectorX, vector_length);
      util::InitializeVector<double>((double *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintVector<double>((double *)HostVectorY, vector_length);
       
      break;
    }

    case 'C': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      
      break;
    }

    case 'H': {
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      
      break;
    }

    case 'Z': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);      
      
      break;
    }

    case 'T': {
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      util::InitializeComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);

      std::cout << "\nVector X of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
      std::cout << "\nVector Y of size " << vector_length << "\n" ;
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);      
      
      break;
    }
  }
  
  //! Allocating Device Memory for Vectors using cudaMalloc()
  cudaStatus = cudaMalloc((void **)&DeviceVectorX, vector_length * sizeof(*HostVectorX));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for X " << std::endl;
    FreeMemory();
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceVectorY, vector_length * sizeof(*HostVectorY));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for Y " << std::endl;
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
  
  //! Copying values of Host vector to Device vector using cublasSetVector()
  status = cublasSetVector(vector_length, sizeof(*HostVectorX), HostVectorX, VECTOR_LEADING_DIMENSION, DeviceVectorX, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector X from host to device failed\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVectorY), HostVectorY, VECTOR_LEADING_DIMENSION, DeviceVectorY, VECTOR_LEADING_DIMENSION);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "Copying vector Y from host to device failed\n";
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
   * API call to  apply Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise 
   * by angle defined by cos(alpha) = c, sin(alpha) = s)
   * The performed operation is \f$ x[k] = c × x[k] + s × y[j] \f$ and \f$ y[j] = −s × x[k] + c × y[j] \f$ 
   * where \f$ k = 1 + (i−1) * incx \f$ and \f$ j = 1 + (i−1) * incy \f$.  
   * Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran \n
   */
  
  switch (mode) {
    case 'S': {
      std::cout << "\nCalling Srot API\n";
      clk_start = clock();

      status = cublasSrot(handle, vector_length, (float *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (float *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (float *)&cosine, (float *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Srot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Srot API call ended\n";
      break;
    }

    case 'D': {
      std::cout << "\nCalling Drot API\n";
      clk_start = clock();

      status = cublasDrot(handle, vector_length, (double *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (double *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (double *)&cosine, (double *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout<< " Drot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Drot API call ended\n";
      break;
    }

    case 'C': {
      std::cout << "\nCalling Crot API\n";
      clk_start = clock();

      status = cublasCrot(handle, vector_length, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (float *)&cosine, (cuComplex *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Crot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Crot API call ended\n";
      break;
    }

    case 'H': {
      std::cout << "\nCalling Csrot API\n";
      clk_start = clock();

      status = cublasCsrot(handle, vector_length, (cuComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (cuComplex *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (float *)&cosine, (float *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Csrot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Csrot API call ended\n";
      break;
    }

    case 'Z': {
      std::cout << "\nCalling Zrot API\n";
      
      clk_start = clock();

      status = cublasZrot(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (double *)&cosine, (cuDoubleComplex *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zrot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zrot API call ended\n";
      break;
    }

    case 'T': {
      std::cout << "\nCalling Zdrot API\n";
      clk_start = clock();

      status = cublasZdrot(handle, vector_length, (cuDoubleComplex *)DeviceVectorX, VECTOR_LEADING_DIMENSION, (cuDoubleComplex *)DeviceVectorY, 
                          VECTOR_LEADING_DIMENSION, (double *)&cosine, (double *)&sine);

      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Zdrot kernel execution error\n";
        FreeMemory();
        return EXIT_FAILURE;
      }

      clk_end = clock();
      std::cout << "Zdrot API call ended\n";
      break;
    }
  }
  
  //! Copy Vector X, holding resultant Vector, from Device to Host using cublasGetVector()
  status = cublasGetVector(vector_length, sizeof (*HostVectorX), DeviceVectorX, VECTOR_LEADING_DIMENSION, HostVectorX, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector x from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  status = cublasGetVector(vector_length, sizeof (*HostVectorY), DeviceVectorY, VECTOR_LEADING_DIMENSION, HostVectorY, VECTOR_LEADING_DIMENSION);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << " Unable to get output vector y from device\n";
    FreeMemory();
    return EXIT_FAILURE;
  }

  switch (mode) {
    case 'S': {  
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintVector<float>((float *)HostVectorX, vector_length);

      std::cout << "\nVector y after " << mode << "rot operation is:\n";
      util::PrintVector<float>((float *)HostVectorY, vector_length);
      break;
    }

    case 'D': {
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintVector<double>((double *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rot operation is:\n";
      util::PrintVector<double>((double *)HostVectorY, vector_length);
      break;
    }

    case 'C': {
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);
      break;
    }

    case 'H': {
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuComplex>((cuComplex *)HostVectorY, vector_length);

      break;
    }

    case 'Z': {
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);
 
      std::cout << "\nVector Y after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
      break;
    }

    case 'T': {
      std::cout << "\nVector X after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorX, vector_length);

      std::cout << "\nVector Y after " << mode << "rot operation is:\n";
      util::PrintComplexVector<cuDoubleComplex>((cuDoubleComplex *)HostVectorY, vector_length);
      break;
    }
  }

  long long total_operations = vector_length;

  //! printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end, total_operations) << "\n\n";

  FreeMemory();

  return EXIT_SUCCESS;
}

void mode_S(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  float sine = (float)sine_real;
  float cosine_f = (float) cosine;


  Rot<float, float, float> Srot(vector_length, sine, cosine_f,  'S' );
  Srot.RotApiCall();
}

void mode_D(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  double sine = sine_real;

  Rot<double, double, double> Drot(vector_length, sine, cosine, 'D');
  Drot.RotApiCall();
}

void mode_C(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  cuComplex sine = {(float)sine_real, (float)sine_imaginary};
  float cosine_f = (float) cosine;

  Rot<cuComplex, cuComplex, float> Crot(vector_length, sine, cosine_f, 'C');
  Crot.RotApiCall(); 
}

void mode_H(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  float sine = (float)sine_real;
  float cosine_f = (float) cosine;

  Rot<cuComplex, float, float> Csrot(vector_length, sine, cosine_f, 'H');
  Csrot.RotApiCall(); 
}

void mode_Z(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  cuDoubleComplex sine = {sine_real, sine_imaginary};

  Rot<cuDoubleComplex, cuDoubleComplex, double> Zrot(vector_length, sine, cosine, 'Z');
  Zrot.RotApiCall(); 
}

void mode_T(int vector_length, double sine_real, double sine_imaginary, double cosine) {
            
  double sine = sine_real;

  Rot<cuDoubleComplex, double, double> Zdrot(vector_length, sine, cosine, 'T');
  Zdrot.RotApiCall(); 
}


void (*cublas_func_ptr[])(int, double, double, double) = {
  mode_S, mode_D, mode_C, mode_H, mode_Z, mode_T
};

int main(int argc, char **argv) {

  int vector_length;
  double sine_real, sine_imaginary;
  double cosine;
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

    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-sine_real"))) 
      sine_real = std::stod(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-sine_imaginary")))
      sine_imaginary = std::stod(argv[loop_count + 1]);

    else if(!(cmd_argument.compare("-cosine"))) 
      cosine = std::stod(argv[loop_count + 1]);   

    else if (!(cmd_argument.compare("-mode")))
      mode = *(argv[loop_count + 1]);
  }
  
  (*cublas_func_ptr[mode_index[mode]])(vector_length, sine_real, sine_imaginary, cosine);
  
  return 0;
}
