%%writefile cublas_utility.h
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 1000 * 1.00) / 100    // to generate random values

#define VECTOR_LEADING_DIMENSION 1

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
  void PrintTriangularMatrix(C* Matrix, int matrix_row, int matrix_col) {
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
  void PrintTriangularComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
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
  void PrintBatchedMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
  void PrintBatchedComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
          std::cout << Matrix[INDEX(row, col, matrix_row) + 
                       batch * matrix_row * matrix_col].x << "+" << 
                       Matrix[INDEX(row, col, matrix_row) + 
                       batch * matrix_row * matrix_col].y << "*I ";
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";
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
  void InitializeTriangularMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        if (row >= col)
          Matrix[INDEX(row, col, matrix_row)] = RANDOM;
      }
    }
  }

  template<class C>
  void InitializeTriangularComplexMatrix(C* Matrix, int matrix_row, int matrix_col) {
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
  void InitializeBatchedMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
  void InitializeBatchedTriangularMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
  void InitializeBatchedTriangularComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
  void InitializeBatchedComplexMatrix(C** Matrix, int matrix_row, int matrix_col, int batch_count) {
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
  void InitializeVector(C* Vector, int vector_length) {
    int index;
    for (index = 0; index < vector_length; index++) {
      Vector[index] = 1;                               
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
    std::cout << "\n";
    for (index = 0; index < vector_length; index++) {
      std::cout << Vector[index] << " "; 
    }
    std::cout << "\n";
  }

  template<class C>
  void PrintComplexVector(C* Vector, int vector_length) {
    int index;
    std::cout << "\n";
    for (index = 0; index < vector_length; index++) {
      std::cout << Vector[index].x << "+" << Vector[index].y << "*I "   << " "; 
    }
    std::cout << "\n";
  }

  template<class C>
  void InitializeDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col, 
	                              int super_diagonals, int sub_diagonals) {
    int super, sub, index;
    int total_values = matrix_row * matrix_col;
    int n =11;

    for (super = 0; super <= super_diagonals; super++) {
      for (index = super; index < total_values; index = index + matrix_col + 1) {
        Matrix[index] = n++;
        if ((index + 1) % matrix_col == 0)
          break;
      }
    }

    for (sub = 1; sub <= sub_diagonals; sub++) {
      for (index = matrix_col * sub; index < total_values; index = index + matrix_col + 1) {
        Matrix[index] = n++;
        if ((index + 1) % matrix_col == 0)
          break;
      }
    }
  }

    template<class C>
  void InitializeComplexDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col, 
	                                     int super_diagonals, int sub_diagonals) {
    int super, sub, index;
    int total_values = matrix_row * matrix_col;

    for (super = 0; super <= super_diagonals; super++) {
      for (index = super; index < total_values; index = index + matrix_col + 1) {
        Matrix[index].x = RANDOM;
        Matrix[index].y = RANDOM;

        if ((index + 1) % matrix_col == 0)
          break;
      }
    }

    for (sub = 1; sub <= sub_diagonals; sub++) {
      for (index = matrix_col * sub; index < total_values; index = index + matrix_col + 1) {
        Matrix[index].x = RANDOM;
        Matrix[index].y = RANDOM;
        if ((index + 1) % matrix_col == 0)
          break;
      }
    }
  }

  template<class C>
  void PrintDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      std::cout << "\n";
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(col, row, matrix_col)] << "   ";
      }
    }
    std::cout << "\n";
  }

  template<class C>
  void PrintComplexDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col) {
    int row, col;
    for (row = 0; row < matrix_row; row++) {
      std::cout << "\n";
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(col, row, matrix_col)].x << "+" 
                  << Matrix[INDEX(col, row, matrix_col)].y << "*I  ";
      }
    }
    std::cout << "\n";
  }



  template<class C>
  void InitializeSymmetricPackedMatrix(C* Matrix, int matrix_size) {
    int index;
    for (index = 0; index < matrix_size; index++) {
      Matrix[index] = RANDOM;
    }
  }
 
 template<class C>
  void InitializeSymmetricPackedComplexMatrix(C* Matrix, int matrix_size) {
    int index;
    for (index = 0; index < matrix_size; index++) {
      Matrix[index].x = RANDOM;
      Matrix[index].y = RANDOM;
    }
  }


  template<class C>
  void PrintSymmetricPackedUpperMatrix(C* Matrix, int matrix_row, int matrix_size) {
    int row, col, jump;
    int row_elments = 1;
    int index;
    for (row = 0; row < matrix_row; row++) {
      jump = matrix_row - 1;
      index = row;
      for(col = 0; col < row_elments; col++) {
        std::cout << Matrix[index] << "  ";
        index += jump;
        jump--;
      }
      std::cout << std::endl;
      row_elments++;
    }
  }
  
  template<class C>
  void PrintSymmetricPackedUpperComplexMatrix(C* Matrix, int matrix_row, int matrix_size) {
    int row, col, jump;
    int row_elments = 1;
    int index;
    for (row = 0; row < matrix_row; row++) {
      jump = matrix_row - 1;
      index = row;
      for(col = 0; col < row_elments; col++) {
        std::cout << Matrix[index].x << "+" 
                  << Matrix[index].y << "*I  ";

        index += jump;
        jump--;
      }
      std::cout << std::endl;
      row_elments++;
    }
  }
  

template<class C>
void InitializeTriangularBandedMatrix(C *Matrix, int matrix_row, int sub_diagonals) {
  int row, col;
  int index = 11;
  for (row = 0; row <= sub_diagonals && row < matrix_row; row++) {
    for (col = 0; col < matrix_row - row; col++) {
      Matrix[INDEX(row, col, matrix_row)] = index++;
    }
  }
}

template<class C>
void InitializeTriangularBandedComplexMatrix(C *Matrix, int matrix_row, int sub_diagonals) {
  int row, col;
  int index = 11;
  for (row = 0; row <= sub_diagonals && row < matrix_row; row++) {
    for (col = 0; col < matrix_row - row; col++) {
      Matrix[INDEX(row, col, matrix_row)].x = index++;
      Matrix[INDEX(row, col, matrix_row)].y = index++;
    }
  }
}

/*
template<class C>
void Print1(C *a, int n) {
  int row, col;
  for (row = 0; row < n; row++) {
    for (col = 0; col < n; col++) {
      std::cout << a[col * n + row] << " ";
    }
    std::cout << std::endl;
  }
}  */

template<class C>
void PrintTriangularBandedMatrix(C *Matrix, int matrix_row) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_row; col++) {
      if (row >= col)
        std::cout << Matrix[col * (matrix_row - 1) + row] << " ";
    }
    std::cout << std::endl;
  }
}

template<class C>
void PrintTriangularBandedComplexMatrix(C *Matrix, int matrix_row) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_row; col++) {
      if (row >= col)
        std::cout << Matrix[col * (matrix_row - 1) + row].x << "+" << Matrix[col * (matrix_row - 1) + row].y << "*I ";
    }
    std::cout << std::endl;
  }
}
 
}  // end of namespace util

