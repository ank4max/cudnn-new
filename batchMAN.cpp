/**
 * Copyright 2020-2021 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    cudnn_batchnormal_test.cpp
 * @brief   cudnn batch normalization API testing
 *
 * @author  ashish(CAI)
 * @date    2021-12-10
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>

void print_array(float *array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{    
    // CAI_AG - Reading values for input parameters using command line arguments 
    for (int i = 0;i < 5; i++)
        std::cout << argv[i] << std::endl;
    
    int n, c, h, w;
    std::string a;

    for (int i = 1; i < 5; i++) {
        int len = sizeof(argv[i]);
        if (argv[i][1] == 'n')
          n = atoi(argv[i] + 2);
        else if (argv[i][1] == 'c')
          c = atoi(argv[i] + 2);
        else if (argv[i][1] == 'h')
          h = atoi(argv[i] + 2);
        else if (argv[i][1] == 'w')
          w = atoi(argv[i] + 2);
 
   }

    // CAI_AG - Generating random input_data 
    int size = n*c*h*w;
    int input_data[size];
    for (int i = 0; i < size; i++)
      input_data[i] = rand() % 255;
 
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
  
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;

    // CAI_AG - setting parameters for batchnormal API
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
    float one = 1.0;
    float zero = 0.0;

    int size_bytes = size * sizeof(float);
    int mean_size = c;
    int mean_size_bytes = mean_size * sizeof(float);

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    std::cout << "Creating descripter" << std::endl;
  
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc, format, dtype, n, c, h, w);

    float *x, *y, *dy, *dx;
    cudaMallocManaged(&x, size_bytes);
    cudaMallocManaged(&y, size_bytes);
    cudaMallocManaged(&dy, size_bytes);
    cudaMallocManaged(&dx, size_bytes);

    // initializing data    
    for (int i = 0; i < size; i++) {
      x[i] = input_data[i];
    }
    std::cout << "Original array: " << std::endl; 
    for(int i=0;i<size;i++) 
        std::cout << x[i] << " ";
    std::cout << std::endl;

    // create activation function descriptor
    float alpha[c] = {1};
    float beta[c] = {0.0};

    cudnnTensorDescriptor_t mean_descriptor;
    cudnnCreateTensorDescriptor(&mean_descriptor);
    cudnnSetTensor4dDescriptor(mean_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/c,
                                        /*image_height=*/1,
                                        /*image_width=*/1);
    
    float *scale, *offset, *dscale, *doffset;
    float *running_mean, *running_var;
    float *saved_mean, *saved_inv_var;
    cudaMallocManaged(&scale, mean_size_bytes);
    cudaMallocManaged(&offset, mean_size_bytes);
    cudaMallocManaged(&dscale, mean_size_bytes);
    cudaMallocManaged(&doffset, mean_size_bytes);
    cudaMallocManaged(&running_mean, mean_size_bytes);
    cudaMallocManaged(&running_var, mean_size_bytes);
    cudaMallocManaged(&saved_mean, mean_size_bytes);
    cudaMallocManaged(&saved_inv_var, mean_size_bytes);

    // initialize scale, offset, running_mean, running_var
    for (int i = 0; i < mean_size; i++) {
      scale[i] = 1.0;
      offset[i] = 1.0;
      running_mean[i] = 1.0;
      running_var[i] = 1.0;
    }

    cudnnActivationDescriptor_t activation_desc;
    cudnnCreateActivationDescriptor(&activation_desc);
    cudnnSetActivationDescriptor(activation_desc,
                                          CUDNN_ACTIVATION_IDENTITY,
                                          CUDNN_PROPAGATE_NAN, 0.0);

    size_t workspace_size_bytes = 0;
    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
    /*xDesc=*/x_desc, /*zDesc=*/NULL, /*yDesc=*/y_desc,
    /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
    /*activationDesc=*/activation_desc,
    /*sizeInBytes=*/&workspace_size_bytes);
 
    void *workspace = nullptr;
    if (workspace_size_bytes > 0) {
      cudaMalloc(&workspace, workspace_size_bytes);
    }

    clock_t start, stop;
    start=clock();
  
    size_t reserve_space_size_bytes = 0;
    cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    /*handle=*/handle_, /*mode=*/mode, /*bnOps=*/bn_ops,
    /*activationDesc=*/activation_desc, /*xDesc=*/x_desc,
    /*sizeInBytes=*/&reserve_space_size_bytes);
    char *reserve_space;
    cudaMalloc(&reserve_space, reserve_space_size_bytes);

    cudnnBatchNormalizationForwardTraining(
    /*handle=*/handle_,
    /*mode=*/mode,
    /**alpha=*/&one,
    /**beta=*/&zero,
    /*xDesc=*/x_desc,
    /**x=*/x,
    /*yDesc=*/y_desc,
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
	std::cout<<(size)<<std::endl;
	  for(int i=0;i<5;i++)
	  std::cout <<"Input n*c*h*w........"<<size*(i+1)<< "...................latancy is "<<time_taken<< "...................Throughput "<<(i+1)* (1e-9*flopsCoef*size)/(time_taken)<<"\n";
    
    cudaDeviceSynchronize();

    print_array(y, size, "output: ");


    cudaFree(x);
    cudaFree(y);
    cudaFree(dy);
    cudaFree(dx);
    cudaFree(scale);
    cudaFree(offset);
    cudaFree(dscale);
    cudaFree(doffset);
    cudaFree(running_mean);
    cudaFree(running_var);
    cudaFree(saved_mean);
    cudaFree(saved_inv_var);
    cudaFree(workspace);
    cudaFree(reserve_space);

    cudnnDestroy(handle_);
    return 0;
}
