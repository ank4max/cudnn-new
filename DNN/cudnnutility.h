%%writefile cudnn_utility.h
#include <iostream>

namespace Util {
  void PrintTensor(float *tensor, int n, int c, int h, int w) {
    int a = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < c; ++j) {
        std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
        for (int k = 0; k < h; ++k) {
          for (int l = 0; l < w; ++l) {
            std::cout << std::setw(6) << std::right << " " << tensor[a];
            ++a;
          }
          std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl;
  }


  void ActivationPrint(float *Tensor, int size) {
    
    for (int i = 0; i < size; i++) {
      std::cout << Tensor[i] << " ";
    }
    std::cout << std::endl;
      
  }

  void InitializeInputTensor(float *Tensor, int size) {
    for (int i = 0; i < size; i++) {
      Tensor[i] = rand() % 255;
    }
  }

  void ActivationInitializer(float *Tensor, int size) {
    for (int i = 0; i < size; i++) {
      Tensor[i] = rand() % 10;
    }
  } 

  
  void InitializeFilterTensor(float *Tensor, int size) {
    for (int i = 0; i < size; i++) {
      Tensor[i] = rand() % 2;
    }
  }
}
