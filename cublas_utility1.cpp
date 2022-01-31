%%writefile cublas_utility.h
#include <iostream>

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values

namespace util {
  template<class C>
  void PrintMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      std::cout << "\n";
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
      }
    }
    std::cout << "\n";
  }

  template<class C>
  void PrintComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
      }
      std::cout << "\n";
    } 
  }
  
  template<class C>
  void PrintSymmetricMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row , col;  
    for (row = 0; row < matrix_row; row++) {                                              
      for (col = 0; col < matrix_col; col++) {
        if (row >= col) {                                                  
          std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
        }
      }
      std::cout << "\n";                                                                                    
    }                                                                               
  }

  template<class C>
  void PrintSymmetricComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        if (row >= col) {
          std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
        }        
      }
      std::cout << "\n";
    } 
  }

  template<class C>
  void InitializeMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row , col;  
    for (row = 0; row < matrix_row; row++) {                                              
      for (col = 0; col < matrix_col; col++) {                                                   
        Matrix[INDEX(row, col, matrix_row)] = RANDOM;                                      
      }                                                                                    
    }                                                                               
  }

  template<class C>
  void InitializeComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;  
    for (col = 0; col < matrix_col; col++) {           
      for (row = 0; row < matrix_row; row++) {                      
        Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
        Matrix[INDEX(row, col, matrix_row)].y = RANDOM;              
      }
    }
  }

  template<class C>
  void InitializeSymmetricMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row , col;  
    for (row = 0; row < matrix_row; row++) {                                              
      for (col = 0; col < matrix_col; col++) {
        if (row >= col) {                                                  
          Matrix[INDEX(row, col, matrix_row)] = RANDOM;
        }
      }                                                                                    
    }                                                                               
  }
  
  template<class C>
  void InitializeSymmetricComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;  
    for (col = 0; col < matrix_col; col++) {           
      for (row = 0; row < matrix_row; row++) {
        if (row >= col) {                      
          Matrix[INDEX(row, col, matrix_row)].x = RANDOM;             
          Matrix[INDEX(row, col, matrix_row)].y = RANDOM;
        }              
      }
    }
  }

  template<class C>
  void InitializeVector(C* Vector, int vector_length) {
    int index;
    for (index = 0; index < vector_length; index++) {
        Vector[index] = RANDOM;                               
    } 

  }

  template<class C>
  void InitializeComplexVector(C* Vector, int vector_length) {
    int index;
    for (index = 0; index < vector_length; index++) {
      Vector[index].x = RANDOM;  
      Vector[index].y = RANDOM;                             
    }

  }

  template<class C>
  void PrintVector(C* Vector, int vector_length) {
    int index;
    for (index = 0; index < vector_length; index++) {
        std::cout << Vector[index] << " "; 
      }
  }

  template<class C>
  void PrintComplexVector(C* Vector, int vector_length) {
    int index;
    for (index = 0; index < vector_length; index++) {
        std::cout << Vector[index].x << "+" << Vector[index].y << "*I "   << " "; 
      }
  }

  template<class C>
  void InitializeStridedMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;

    for (batch = 0; batch < batch_count; batch++) {
      Matrix[batch] = new C[matrix_row * matrix_col];
      for (col = 0; col < matrix_col; col++) {
        for (row = 0; row < matrix_row; row++) {
          Matrix[batch][INDEX(row, col, matrix_row)] = RANDOM;
      }
    }
  }

}

template<class C>
  void InitializeStridedTriangularMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;

  for (batch = 0; batch < batch_count; batch++) {
    Matrix[batch] = new C[matrix_row * matrix_col];
    for (col = 0; col < matrix_col; col++) {
      for (row = 0; row < matrix_row; row++) {
        if (row >= col)
          Matrix[batch][INDEX(row, col, matrix_row)] = RANDOM;
        else
          Matrix[batch][INDEX(row, col, matrix_row)] = 0.0;
      }
    }
  }

}

template<class C>
  void InitializeStridedTriangularComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;

  for (batch = 0; batch < batch_count; batch++) {
    Matrix[batch] = new C[matrix_row * matrix_col];
    for (col = 0; col < matrix_col; col++) {
      for (row = 0; row < matrix_row; row++) {
        if (row >= col) {
          Matrix[batch][INDEX(row, col, matrix_row)].x = RANDOM;
          Matrix[batch][INDEX(row, col, matrix_row)].y = RANDOM;
        }
        else {
          Matrix[batch][INDEX(row, col, matrix_row)].x = 0.0;
          Matrix[batch][INDEX(row, col, matrix_row)].y = 0.0;
        }
      }
    }
  }

}



 template<class C>
  void InitializeStridedComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;

    for (batch = 0; batch < batch_count; batch++) {
      Matrix[batch] = new C[matrix_row * matrix_col];
      for (col = 0; col < matrix_col; col++) {
        for (row = 0; row < matrix_row; row++) {
          Matrix[batch][INDEX(row, col, matrix_row)].x = RANDOM;
          Matrix[batch][INDEX(row, col, matrix_row)].y = RANDOM;
      }
    }
  }

}

template<class C>
void InitializeStridedBatchedMatrix(C* Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;
    for (batch = 0; batch < batch_count; batch++) {
      for (row = 0; row < matrix_row; row++) {
        for (col = 0; col < matrix_col; col++) {
          Matrix[INDEX(row, col, matrix_row) + batch * matrix_col * matrix_row] = RANDOM;
      }
    }
  }

}

template<class C>
void InitializeStridedBatchedComplexMatrix(C* Matrix, int matrix_row, int matrix_col, int batch_count) {
    int row, col, batch;
    for (batch = 0; batch < batch_count; batch++) {
      for (row = 0; row < matrix_row; row++) {
        for (col = 0; col < matrix_col; col++) {
          Matrix[INDEX(row, col, matrix_row) + batch * matrix_col * matrix_row].x = RANDOM;
          Matrix[INDEX(row, col, matrix_row) + batch * matrix_col * matrix_row].y = RANDOM;
      }
    }
  }

}







template<class C>  
void PrintStridedMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
  int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[batch][INDEX(row, col, matrix_row)] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}


template<class C>  
void PrintStridedComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
  int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[batch][INDEX(row, col, matrix_row)].x << "+" << Matrix[batch][INDEX(row, col, matrix_row)].y << "*I ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

template<class C>  
void PrintStridedBatchedMatrix(C* Matrix, int matrix_row, int matrix_col, int batch_count) {  

 int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(row, col, matrix_row) + batch * matrix_row * matrix_col] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

template<class C>  
void PrintStridedBatchedComplexMatrix(C* Matrix, int matrix_row, int matrix_col, int batch_count) {  

 int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(row, col, matrix_row) + batch * matrix_row * matrix_col].x << "+" << Matrix[INDEX(row, col, matrix_row) + batch * matrix_row * matrix_col].y << "*I ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

}
