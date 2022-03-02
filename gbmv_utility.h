%%writefile cublas_utility.h
#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))    // for getting index values matrices
#define RANDOM (rand() % 1000 * 1.00) / 100    // to generate random values

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
  void InitializeDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col, 
	                          int super_diagonals, int sub_diagonals) {
  int super, sub, index;
  int total_values = matrix_row * matrix_col;
  
  for (super = 0; super <= super_diagonals; super++) {
	for (index = super; index < total_values; index = index + matrix_col + 1) {
	  Matrix[index] = RANDOM;
	}
  }

  for (sub = 1; sub <= sub_diagonals; sub++) {
	for (index = matrix_col * sub; index < total_values; index = index + matrix_col + 1) {
	  Matrix[index] = RANDOM;
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
  void PrintDiagonalMatrix(C* Matrix, int matrix_row, int matrix_col, int super_diagonals) {
    int row , col;
    int ind =11;
    // highest superdiagonal 11 ,12 ,13 ,14 in first row ,
    // starting from i=ku
    for(row = super_diagonals; row < matrix_col; row++) {
        std::cout << Matrix[ INDEX (0 , row, matrix_row )] ; 
    }
    //std::cout << std::endl;

    // next superdiagonal 15 ,16 ,17 ,18 ,19 in next row ,
    // starting from i=ku -1
    for(row = super_diagonals -1; row < matrix_col; row++) {
        std::cout << Matrix[ INDEX (1 , row, matrix_row )]; 
    }
    //std::cout << std::endl;

    // main diagonal 20 ,21 ,22 ,23 ,24 in row ku , starting from i=0
    for(row = 0; row < matrix_col -1; row++)  {
        Matrix[ INDEX( super_diagonals, row, matrix_row )] = ( float )ind ++;
    }    
    // subdiagonal 25 ,26 ,27 ,28 in ku +1 row , starting from i=0
    for(row = 0; row < matrix_col -2; row++) { 
        Matrix[ INDEX (super_diagonals +1 , row, matrix_row )] = ( float )ind ++;
  }
}


}  // end of namespace util


