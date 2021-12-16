#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include<bits/stdc++.h>
#include <time.h>

using namespace std;


/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN.
 **/

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
int main(int argc, char** argv)
{    
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

  clock_t start, stop;
  start=clock();
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

stop=clock();
 double flopsCoef = 2.0;
	double time_taken=double(stop-start)/double(CLOCKS_PER_SEC);
for(int i=0;i<5;i++)
{  std::cout <<"Input n*c*h*w........"<<x_size*(i+1)<< "...................latancy is "<<std::fixed<<time_taken<<std::endl;
	std::cout<<" Throughput  "<<(i+1)* (1e-9*flopsCoef*x_size)/(time_taken)<<"\n";
 }
  checkCUDA(cudaDeviceSynchronize());

  //print_array(y, x_size, "output: ");
  
  
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
