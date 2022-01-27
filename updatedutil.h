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
      HostVecX[index].x = RANDOM;  
      HostVecX[index].y = RANDOM;                             
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
  
}
