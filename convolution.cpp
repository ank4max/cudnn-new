%%writefile max.cpp

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>



void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost);
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}



int main(int argc, char** argv) {    
  // Reading values for input parameters using command line arguments 
  std::cout << "\n\n" << std::endl;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }    
  int batch, channel, height, width;
  std::string mode_set;

 //! reading cmd line arguments and initializing the required parameters
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);

    if (!(cmd_argument.compare("-batch")))
      batch = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-channel")))
      channel = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-height")))
      height = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-width")))
      width = std::stod(argv[loop_count + 1]);
  }

  // Generating random input_data 
  int size = batch * channel * height * width;
  float InputData[size];
  for (int i = 0; i < size; i++) {
    InputData[i] = rand() % 255;
  }
  int num_GPUs;
  cudaError_t cudaStatus;
  cudnnStatus_t status;
  
  cudaStatus = cudaGetDeviceCount(&num_GPUs);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU count error");
    return EXIT_FAILURE;   
  }
  std::cout << "Found " << num_GPUs << " GPUs." << std::endl;
  
  //setting device
  cudaStatus = cudaSetDevice(0);
  if( cudaStatus != cudaSuccess) {
    printf(" GPU set device  error");
    return EXIT_FAILURE;   
  }

  int device; 
  struct cudaDeviceProp devProp;
  cudaStatus = cudaGetDevice(&device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device read  error");
    return EXIT_FAILURE;   
  }
  
  cudaStatus = cudaGetDeviceProperties(&devProp, device);
  if( cudaStatus != cudaSuccess) {
    printf(" Device properties  error\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

  cudnnHandle_t handle;
  status = cudnnCreate(&handle);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to initialize handle\n");
    return EXIT_FAILURE;   
  }
  std::cout << "Created cuDNN handle" << std::endl;
  
  cudnnTensorDescriptor_t input_desc;
  status = cudnnCreateTensorDescriptor(&input_desc);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" create input descriptor error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set tensor descriptor error for input\n");
    return EXIT_FAILURE;   
  }
  
  float *in_data;
  cudaStatus = cudaMallocManaged(&in_data, size * sizeof(float));

  if( cudaStatus != cudaSuccess) {
    printf(" Device Memory allocation error for in_data\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaMemcpy(in_data, InputData, size * sizeof(float), cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf(" failed to copy input data to device\n");
    return EXIT_FAILURE;   
  }

  
  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  status  = cudnnCreateFilterDescriptor(&filt_desc); 
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Create filter Descriptor error\n");
    return EXIT_FAILURE;   
  }



  status = cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set filter Descriptor error\n");
    return EXIT_FAILURE;   
  }
  
  int size1 = filt_k * filt_c * filt_h * filt_w ;
  float *filt_data;
  cudaStatus = cudaMallocManaged(&filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float));

  if( cudaStatus != cudaSuccess) {
    printf(" Device memory allocation error for filt_data\n");
    return EXIT_FAILURE;   
  }

  float fill_data[size1];
  for (int i = 0; i < size1; i++) {
    if(i%2 == 0) {
       fill_data[i] = 1; 
    }
    else {
        fill_data[i] = 0;
    }
  }

  cudaStatus = cudaMemcpy(filt_data, fill_data, size1 * sizeof(float), cudaMemcpyHostToDevice);
  if( cudaStatus != cudaSuccess) {
    printf(" failed to copy input data to device\n");
    return EXIT_FAILURE;   
  }

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  status = cudnnCreateConvolutionDescriptor(&conv_desc);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Create conv Descriptor error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set Convolution Descriptor error\n");
    return EXIT_FAILURE;   
  }

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  status = cudnnGetConvolution2dForwardOutputDim(
        conv_desc, input_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w);

  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Set GetConvolution2dForwardOutputDim error\n");
    return EXIT_FAILURE;   
  }

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  
 cudnnTensorDescriptor_t out_desc;
  cudnnCreateTensorDescriptor(&out_desc);
  status = cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Output Tensor descriptor error\n");
    return EXIT_FAILURE;   
  }

  float *out_data;
  cudaStatus = cudaMallocManaged(&out_data, out_n * out_c * out_h * out_w * sizeof(float));
         

    if( cudaStatus != cudaSuccess) {
    printf(" device allocation for  out_data\n");
    return EXIT_FAILURE;   
  }   
  
  //cudaStatus = cudaMemset(out_data, 0, out_n * out_c * out_h * out_w * sizeof(float));
  if( cudaStatus != cudaSuccess) {
    printf(" Memset error for out_data\n");
    return EXIT_FAILURE;   
  }   
  

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  status = cudnnGetConvolutionForwardAlgorithm(
        handle,
        input_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Forward Algorithm  error\n");
    return EXIT_FAILURE;   
  }


  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  status = cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);
  
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Convolution Forward Workspace size error\n");
    return EXIT_FAILURE;   
  }

  float *ws_data;
  cudaStatus = cudaMallocManaged(&ws_data, ws_size);
  if( cudaStatus != cudaSuccess) {
    printf(" device allocation failed for ws_size\n");
    return EXIT_FAILURE;   
  } 


  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  
  
//the convolution
const float alpha = 1, beta = 0;

status = cudnnConvolutionForward(
      handle,
      &alpha, input_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data);

if( status != CUDNN_STATUS_SUCCESS) {
    printf(" API faied to execute\n");
    return EXIT_FAILURE;   
  }


// results
  std::cout << "in_data:" << std::endl;
  print(in_data, batch, channel, height, width);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  cudaStatus = cudaFree(ws_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(out_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyTensorDescriptor(out_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy output Descriptor\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyConvolutionDescriptor(conv_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy convolution Descriptor\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(filt_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyFilterDescriptor(filt_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy Filter Descriptor\n");
    return EXIT_FAILURE;   
  }

  cudaStatus = cudaFree(in_data);
  if( cudaStatus != cudaSuccess) {
    printf(" Device memmory deallocation error\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroyTensorDescriptor(input_desc);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy input Descriptor\n");
    return EXIT_FAILURE;   
  }

  status = cudnnDestroy(handle);
  if( status != CUDNN_STATUS_SUCCESS) {
    printf(" Unable to Destroy handle\n");
    return EXIT_FAILURE;   
  }
  return 0;
}
