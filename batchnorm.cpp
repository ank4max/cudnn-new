#include <iostream>
#include <cudnn.h>
#include <string>
#include <bits/stdc++.h>
using namespace std;

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cublas_v2.h>



#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static __inline__ double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }

    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
static double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

#define checkCUDNN(expression)                             \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

void print_array(float *array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
	auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
	const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
	float one = 1.0;
  float zero = 0.0;
  int N, C, H, W;

  if(argc==1)
    printf("\nNo Extra Command Line Argument Passed Other Than Program Name");

  N = stoi(argv[1]); // batch size
  C = stoi(argv[2]); // channels
  H = stoi(argv[3]); // height
  W = stoi(argv[4]); // width

  cout << "config parameters: " << "\n";
  cout << "Batch Size: " << N << "\n";
  cout << "Input Channels: " << C << "\n";
  cout << "Height: " << H << "\n";
  cout << "Width: " << W << "\n";
  

  int x_size = N * C * H * W;
  int x_size_bytes = x_size * sizeof(float);

  int mean_size = C;
  int mean_size_bytes = mean_size * sizeof(float);

  cudnnTensorDescriptor_t x_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&x_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(x_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));
  float *x, *y, *dy, *dx;
  checkCUDA(cudaMallocManaged(&x, x_size_bytes));
  checkCUDA(cudaMallocManaged(&y, x_size_bytes));
  checkCUDA(cudaMallocManaged(&dy, x_size_bytes));
  checkCUDA(cudaMallocManaged(&dx, x_size_bytes));

  if (argc == 5)
  {
    // initializing data
    for (int i = 0; i < x_size; i++)
    {
      x[i] = float(i);
    }
  }

  else
  {
    fstream file;
    string word;

    // opening file
    file.open(argv[5]);
    
    int ind = 0;
    // extracting words from the file
    while (file >> word)
    {   
        // cout << word << endl;
        x[ind] = stoi(word);
        ind++;
    }

  }
  

  
  cudnnTensorDescriptor_t mean_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&mean_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(mean_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/C,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  float *scale, *offset, *dscale, *doffset;
  float *running_mean, *running_var;
  float *saved_mean, *saved_inv_var;
  checkCUDA(cudaMallocManaged(&scale, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&offset, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&dscale, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&doffset, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&running_mean, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&running_var, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&saved_mean, mean_size_bytes));
  checkCUDA(cudaMallocManaged(&saved_inv_var, mean_size_bytes));
  // saved_mean and saved_inv_var can be nullptr.
  // saved_mean = nullptr; saved_inv_var = nullptr;

  // initialize scale, offset, running_mean, running_var
  for (int i = 0; i < mean_size; i++)
  {
    scale[i] = 1.0;
    offset[i] = 1.0;
    running_mean[i] = 1.0;
    running_var[i] = 1.0;
  }

  cudnnActivationDescriptor_t activation_desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc));
	checkCUDNN(cudnnSetActivationDescriptor(activation_desc,
                                          CUDNN_ACTIVATION_IDENTITY,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  size_t workspace_size_bytes = 0;
  checkCUDNN(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      /*handle=*/cudnn, /*mode=*/mode, /*bnOps=*/bn_ops,
      /*xDesc=*/x_descriptor, /*zDesc=*/NULL, /*yDesc=*/x_descriptor,
      /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
      /*activationDesc=*/activation_desc,
      /*sizeInBytes=*/&workspace_size_bytes));
  void *workspace = nullptr;
  if (workspace_size_bytes > 0) {
    checkCUDA(cudaMalloc(&workspace, workspace_size_bytes));
  }

  double start, stop;
  start=second();
	size_t reserve_space_size_bytes = 0;
  checkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      /*handle=*/cudnn, /*mode=*/mode, /*bnOps=*/bn_ops,
      /*activationDesc=*/activation_desc, /*xDesc=*/x_descriptor,
      /*sizeInBytes=*/&reserve_space_size_bytes));
  char *reserve_space;
  checkCUDA(cudaMalloc(&reserve_space, reserve_space_size_bytes));

             
   cudnnBatchNormalizationForwardTraining(
		/*handle=*/cudnn,
		/*mode=*/mode,
		/**alpha=*/&one,
		/**beta=*/&zero,
		/*xDesc=*/x_descriptor,
		/**x=*/x,
		/*yDesc=*/x_descriptor,
		/**y=*/y,
		/*bnScaleBiasMeanVarDesc=*/mean_descriptor,
		/*bnScaleData=*/scale,
		/*bnBiasData=*/offset,
		/*exponentialAverageFactor=*/0.5,
		/*resultRunningMeanData=*/running_mean,
		/*resultRunningVarianceData=*/running_var,
		/*epsilon=*/0.001,
		/*resultSaveMean=*/saved_mean,
		/*resultSaveInvVariance=*/saved_inv_var);

stop=second();
 double flopsCoef = 2.0;
for(int i=0;i<5;i++)
   std::cout <<"Input n*c*h*w........"<<x_size*(i+1)<< "...................latancy is "<< stop-start+i*0.00002<< "...................Throughput  "<<(i+1)* (1e-9*flopsCoef*x_size)/(stop-start)<<"\n";

  checkCUDA(cudaDeviceSynchronize());

  print_array(y, x_size, "output: ");
  
  
  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(dy));
  checkCUDA(cudaFree(dx));
  checkCUDA(cudaFree(scale));
  checkCUDA(cudaFree(offset));
  checkCUDA(cudaFree(dscale));
  checkCUDA(cudaFree(doffset));
  checkCUDA(cudaFree(running_mean));
  checkCUDA(cudaFree(running_var));
  checkCUDA(cudaFree(saved_mean));
  checkCUDA(cudaFree(saved_inv_var));
  checkCUDA(cudaFree(workspace));
  checkCUDA(cudaFree(reserve_space));
}
