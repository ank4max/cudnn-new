#include <iostream>
#include <string>
#include "cublas_v2.h"
#include<cuda_runtime.h>
           
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 1000000000 * 1.00) / 10000000    // to generate random values
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 
cudaError_t cudaStatus; 
cublasStatus_t status; 
cublasHandle_t handle;
clock_t clk_start, clk_end;


template <typename T>
class dot {
  private :
    T * HostVecX;             
    T * HostVecY;
    T * DeviceVecX;
    T * DeviceVecY;
    int vector_length ;
    T dot_product;
    int index;
    
  public :
    dot(int i) : vector_length(i) {
      
      std ::cout <<"vector length has been allocated with constructor :" << vector_length<<"\n";
        
    }
    
    int HostAlloc() {
    
      HostVecX = new T[vector_length];
      HostVecY = new T[vector_length]; 
      if (HostVecX == 0) {
        fprintf (stderr, "!!!! Host memory allocation error (vector X)\n");
        return EXIT_FAILURE;
      }
      
      if (HostVecY == 0) {
        fprintf (stderr, "!!!! Host memory allocation error (vector Y)\n");
        return EXIT_FAILURE;
      }
      
      return EXIT_SUCCESS;
    }
    
    
    void SetData() {
    
      // setting up values in X and Y vectors
      // using RANDOM macro to generate random float numbers between 0 - 100
      for (index = 0; index < vector_length; index++) {
        HostVecX[index] = RANDOM;                               
      }

      for (index = 0; index < vector_length; index++) {
        HostVecY[index] = RANDOM; 
      }
    }

    void SetComplexData() {
    
      // setting up values in X and Y vectors
      // using RANDOM macro to generate random float numbers between 0 - 100
      for (index = 0; index < vector_length; index++) {
        HostVecX[index].x = RANDOM;  
        HostVecX[index].y = 0.0f;                             
      }

      for (index = 0; index < vector_length; index++) {
        HostVecY[index].x = RANDOM; 
        HostVecY[index].y = 0.0f;
      }
    }

    void PrintVec() {
    
      // printing the initial values in vector X and vector Y
      std::cout << "\nX vector initially:\n";
      for (index = 0; index < vector_length; index++) {
        std::cout << HostVecX[index] << " "; 
      }
      std::cout << "\n";
  
      std::cout << "\nY vector initially :\n";
      for (index = 0; index < vector_length; index++) {
        std::cout << HostVecY[index] << " "; 
      }
      std::cout << "\n";
      
    }
    
    void PrintComplexVec() {
    
      // printing the initial values in vector X and vector Y
      std::cout << "\nX vector initially:\n";
      for (index = 0; index < vector_length; index++) {
        std::cout << HostVecX[index].x << "+" << HostVecX[index].y << "*I "   << " "; 
      }
      std::cout << "\n";
  
      std::cout << "\nY vector initially :\n";
      for (index = 0; index < vector_length; index++) {
        std::cout << HostVecY[index].x << "+" << HostVecY[index].y << "*I "   << " ";
      }
      std::cout << "\n";
      
    }

    
    int DeviceAlloc() {
    
      cudaStatus = cudaMalloc ((void **) &DeviceVecX, vector_length * sizeof (*HostVecX));
      if( cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for X\n";
        return EXIT_FAILURE;
      }
  
      cudaStatus = cudaMalloc ((void **) &DeviceVecY, vector_length * sizeof (*HostVecY));
      if( cudaStatus != cudaSuccess) {
        std::cout << " The device memory allocation failed for Y\n";
        return EXIT_FAILURE;   
      }
 
      //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
      status = cublasCreate (&handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to initialize handle\n");
        return EXIT_FAILURE;
      }

      status = cublasSetVector (vector_length, sizeof (*HostVecX) , HostVecX, 1, DeviceVecX, 1); 
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to set vector values for X on gpu\n");
        return EXIT_FAILURE;
      }
  
      status = cublasSetVector (vector_length, sizeof (*HostVecY), HostVecY, 1, DeviceVecY, 1); 
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to set vector values for Y on gpu\n");
        return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;

    }
    
    void Output() {
     
      //printing the final result
      std::cout << "\nDot product X.Y is : " << dot_product << "\n";   
       
      // printing latency and throughput of the function
      std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
    
    
    }

    void ComplexOutput() {
     
      //printing the final result
      std::cout << "\nDot product X.Y is : " << dot_product.x << "+" << dot_product.y << "*I "<<"\n";  
       
      // printing latency and throughput of the function
      std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
    
    
    }
    
    int Sdot() {
      HostAlloc();
      SetData();
      PrintVec();
      DeviceAlloc();

      std::cout<<" Using sdot api\n";
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasSdot(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Sdot kernel execution error\n");
        return EXIT_FAILURE;
      }
      Output();
      DeAlloc();
      return EXIT_SUCCESS;
    }
    
    int Ddot() {
      HostAlloc();
      SetData();
      PrintVec();
      DeviceAlloc();
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasDdot(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Ddot kernel execution error\n");
        return EXIT_FAILURE;
      }
      Output();
      DeAlloc();
      return EXIT_SUCCESS;

    }

    int Cdotu() {
      HostAlloc();
      SetComplexData();
      PrintComplexVec();
      DeviceAlloc();
      std::cout<<" using Cdot api\n";
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasCdotu(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Cdot kernel execution error\n");
        return EXIT_FAILURE;
      }
    
      ComplexOutput();
      DeAlloc();
      return EXIT_SUCCESS;


    } 

    int Cdotc() {
      HostAlloc();
      SetComplexData();
      PrintComplexVec();
      DeviceAlloc();
      std::cout<<" using Cdotc api\n";
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasCdotc(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Cdotc kernel execution error\n");
        return EXIT_FAILURE;
      }
    
      ComplexOutput();
      DeAlloc();
      return EXIT_SUCCESS;


    } 

    int Zdotu() {
      HostAlloc();
      SetComplexData();
      PrintComplexVec();
      DeviceAlloc();
      std::cout<<" Using Zdotu api\n";
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasZdotu(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Zdotu kernel execution error\n");
        return EXIT_FAILURE;
      }
    
      ComplexOutput();
      DeAlloc();
      return EXIT_SUCCESS;

    }  

   int Zdotc() {
      HostAlloc();
      SetComplexData();
      PrintComplexVec();
      DeviceAlloc();
      std::cout<<" Using Zdotc api\n";
      clk_start = clock();

      // performing dot product operation and storing result in dot_product variable
      status = cublasZdotc(handle, vector_length, DeviceVecX, 1, DeviceVecY, 1, &dot_product);

      clk_end = clock();
  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Zdotc kernel execution error\n");
        return EXIT_FAILURE;
      }
    
      ComplexOutput();
      DeAlloc();
      return EXIT_SUCCESS;

    }                                     

    int DeAlloc() {
    
      //freeing device memory
      cudaStatus = cudaFree (DeviceVecX);
      if( cudaStatus != cudaSuccess) {
        std::cout << " Memory free error on device for vector X\n";
        return EXIT_FAILURE;
      }
  
      cudaStatus = cudaFree (DeviceVecY);
      if( cudaStatus != cudaSuccess) {
        std::cout << " Memory free error on device for vector Y\n";
        return EXIT_FAILURE;
      }
  
      //destroying cublas context and freeing host memory
      status = cublasDestroy (handle);  
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! Failed to uninitialize handle");
        return EXIT_FAILURE;
      }
  
      delete[] HostVecX; 
      delete[] HostVecY; 

      return EXIT_SUCCESS ;
     }
};


int main ( int argc, char **argv) {
  // initializing size of vector with command line arguement
  
  int vector_length;
  char n;
  
  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;
    
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
    else if (!(cmd_argument.compare("-mode")))
      n = *(argv[loop_count + 1]);
      
  }
  
  if(n=='s') {
   std::cout << "using float templatefor dot function\n";
   dot<float>obj1(vector_length);
   obj1.Sdot();
  }
  
  else if(n=='d') {
    std::cout<< "using double template for dot function\n";
    dot<double>obj2(vector_length);
    obj2.Ddot();
    
  }

  else if(n=='u') {
    std::cout<< "using complex template for dot function\n";
    dot<cuComplex>cublas_cdotu(vector_length);
    cublas_cdotu.Cdotu();
    
  }

  else if(n=='c') {
    std::cout<< "using complex template for dot function\n";
    dot<cuComplex>cublas_cdotc(vector_length);
    cublas_cdotc.Cdotc();

  }
 
 else if(n=='U') {
    std::cout<< "using double complex template for dot function\n";
    dot<cuDoubleComplex>cublas_zdotu(vector_length);
    cublas_zdotu.Zdotu();
 
 }
 else if(n=='C') {
    std::cout<< "using double complex template for dot function\n";
    dot<cuDoubleComplex>cublas_zdotc(vector_length);
    cublas_zdotc.Zdotc();
 
 }

  return 0;
}
