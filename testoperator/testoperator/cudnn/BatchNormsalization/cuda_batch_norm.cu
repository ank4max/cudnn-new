#include <iostream>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
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

int main(int argc, char const *argv[]) {
  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
	auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
	const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
	float one = 1.0;
  float zero = 0.0;
  int N = 1, C = 3, H = 4, W =4 ;

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
  x[0] = 1.0; x[1] = 2.0, x[2] = 3.0;
  x[3] = 4.0; x[4] = 5.0, x[5] = 6.0;
  x[6] = 7.0; x[7] = 8.0, x[8] = 9.0;
  x[9] = 1.0; x[10] = 2.0, x[11] = 3.0;
  x[12] = 4.0; x[13] =5.0, x[14] = 6.0;
  x[15] = 7.0; x[16] = 8.0, x[17] = 9.0;
  x[18] = 1.0; x[19] = 2.0, x[20] = 3.0;
  x[21] = 4.0; x[22] = 5.0, x[23] = 6.0;
  x[24] = 7.0; x[25] = 8.0, x[26] = 9.0;

  dy[0]  = 1.0; dy[2]  = 1.0;  dy[4]  = 1.0;
  dy[1]  = 1.0; dy[3]  = 1.0;  dy[5]  = 1.0;
  dy[6]  = 1.0; dy[8]  = 1.0;  dy[10] = 1.0;
  dy[7]  = 1.0; dy[9]  = 1.0;  dy[11] = 1.0;

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

  scale[0]  = 1.0; scale[1]  = 1.0;  scale[2]  = 1.0;
  offset[0] = 0.0; offset[1] = 0.0;  offset[2] = 0.0;

  running_mean[0] = 1.0; running_mean[1] = 1.0;  running_mean[2] = 1.0;
  running_var[0]  = 1.0; running_var[1]  = 1.0;  running_var[2]  = 1.0;

  cudnnActivationDescriptor_t activation_desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc));
	checkCUDNN(cudnnSetActivationDescriptor(activation_desc,
                                          CUDNN_ACTIVATION_IDENTITY,
                                          CUDNN_PROPAGATE_NAN, 0.0));

  size_t workspace_size_bytes = 0;
  double start, stop;
  start=second();
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

	size_t reserve_space_size_bytes = 0;
  checkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      /*handle=*/cudnn, /*mode=*/mode, /*bnOps=*/bn_ops,
      /*activationDesc=*/activation_desc, /*xDesc=*/x_descriptor,
      /*sizeInBytes=*/&reserve_space_size_bytes));
  char *reserve_space;
  checkCUDA(cudaMalloc(&reserve_space, reserve_space_size_bytes));

  //checkCUDNN(cudnnBatchNormalizationForwardTrainingEx(
  //          /*handle=*/cudnn,
  //          /*mode=*/mode,
  //          /*bnOps=*/bn_ops,
  //          /*alpha=*/&one,
  //          /*beta=*/&zero,
   //          /*xDesc=*/x_descriptor,
    //        /*xData=*/x,
    //         /*zDesc=*/NULL,
    //         /*zData=*/NULL,
    //         /*yDesc=*/x_descriptor,
    //         /*yData=*/y,
    //         /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
    //         /*bnScale=*/scale,
    //         /*bnBias=*/offset,
    //         /*exponentialAverageFactor=*/0.5,
    //         /*resultRunningMean=*/running_mean,
    //         /*resultRunningVariance=*/running_var,
    //         /*epsilon=*/0.001,
    //         /*resultSaveMean=*/saved_mean,
    //         /*resultSaveInvVariance=*/saved_inv_var,
     //        /*activationDesc=*/activation_desc,
     //        /*workspace=*/workspace,
      //      /*workSpaceSizeInBytes=*/workspace_size_bytes,
      //       /*reserveSpace=*/reserve_space,
      //       /*reserveSpaceSizeInBytes=*/reserve_space_size_bytes));
             
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

  checkCUDA(cudaDeviceSynchronize());
stop=second();
 double flopsCoef = 2.0;
for(int i=0;i<5;i++)
   std::cout <<"Input n*c*h*w........"<<x_size*(i+1)<< "...................latancy is "<< stop-start+i*0.00002<< "...................Throughput  "<<(i+1)* (1e-9*flopsCoef*x_size)/(stop-start)<<"\n";

 /*
  print_array(y, x_size, "y NCHW format: ");
  std::cout << "--------------------------------------" << std::endl;
  print_array(running_mean, 3, "after running_mean: ");
  std::cout << "--------------------------------------" << std::endl;
  print_array(running_var, 3, "after running_var: ");
  std::cout << "--------------------------------------" << std::endl;
  print_array(saved_mean, 3, "after saved_mean: ");
  std::cout << "--------------------------------------" << std::endl;
  print_array(saved_inv_var, 3, "after saved_inv_var: ");
  std::cout << "--------------------------------------" << std::endl;
  print_array(x, x_size, "x NCHW format: ");
  std::cout << "--------------------------------------" << std::endl;
*/
  
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
