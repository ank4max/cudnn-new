%%writefile cudnn_utility.h
#include <iostream>

#define SPACING 6 
#define RANDOM_PIXEL rand() % 256
#define RANDOM_KERNEL rand() % 2
#define RANDOM_ACTIVATION rand() % 10
#define RANDOM_NORMALIZED (float)rand() / RAND_MAX
#define ALPHA_INITIAL_VALUE 1.0
#define BETA_INITIAL_VALUE 0.0
#define ZERO_INITIAL_VALUE 0.0
#define INITIAL_VALUE 1.0
#define RELU_CLIPPING_THREASHOLD 0.0

namespace Util {
  void PrintTensor(float *Tensor, int batch, int channel, int height, int width) {
    int index = 0;
    for (int n = 0; n < batch; ++n) {
      for (int c = 0; c < channel; ++c) {
        std::cout << "batch = " << n << ", channel = " << c << ": " << std::endl;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            std::cout << std::setw(SPACING) << std::right << " " << Tensor[index];
            ++index;
          }
          std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl;
  }

  void InitializeInputTensor(float *Tensor, int size) {
    for (int index = 0; index < size; index++) {
      Tensor[index] = RANDOM_PIXEL;
    }
  } 
  
  void InitializeFilterTensor(float *Tensor, int size) {
    for (int index = 0; index < size; index++) {
      Tensor[index] = RANDOM_KERNEL;
    }
  }

  void InitializeActivationTensor(float *Tensor, int size) {
    for (int index = 0; index < size; index++) {
      Tensor[index] = RANDOM_ACTIVATION;
    }
  }

  void InitializeNormalizedInputTensor(float *Tensor, int size) {
    for (int index = 0; index < size; index++) {
      Tensor[index] = RANDOM_NORMALIZED;
    }
  }
}
