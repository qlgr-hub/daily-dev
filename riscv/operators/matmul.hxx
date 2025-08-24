#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>


template<typename T>
class Matrix2D {
  uint64_t __rows;
  uint64_t __cols;
  std::unique_ptr<std::unique_ptr<T[]>[]> __data;

public:
  Matrix2D(uint64_t rows, uint64_t cols) : __rows{rows}, __cols{cols} {
    __data = std::make_unique<std::unique_ptr<T[]>[]>(__rows);
    for (uint64_t i = 0; i < __rows; ++i) {
      __data[i] = std::make_unique<T[]>(__cols);
    }
  }

  Matrix2D(const Matrix2D&) = delete;
  Matrix2D& operator=(const Matrix2D&) = delete;
  Matrix2D(Matrix2D&&) noexcept = default;
  Matrix2D& operator=(Matrix2D&&) noexcept = default;

public:
  T* operator[](uint64_t row) {
    return __data[row].get();
  }

  const T* operator[](uint64_t row) const {
    return __data[row].get();
  }

  T& at(uint64_t row, uint64_t col) {
    if (row >= __rows || col >= __cols) {
      throw std::out_of_range("Matrix2D index out of bounds");
    }
    return __data[row][col];
  }

  const T& at(uint64_t row, uint64_t col) const {
    if (row >= __rows || col >= __cols) {
      throw std::out_of_range("Matrix2D index out of bounds");
    }
    return __data[row][col];
  }

  uint64_t rows() const { return __rows; }
  uint64_t cols() const { return __cols; }
};

enum class FType {
  base,
  rvv
};

template <typename T, FType FT>
void matmul(const Matrix2D<T>& A, const Matrix2D<T>& B, Matrix2D<T>& C) {
  assert(A.cols() == B.rows());
  assert(A.rows() == C.rows());
  assert(B.cols() == C.cols());

  if constexpr (FT == FType::base) {
    for (uint64_t m = 0; m < A.rows(); ++m) {
      for (uint64_t n = 0; n < B.cols(); ++n) {
        for (uint64_t k = 0; k < A.cols(); ++k) {
          C[m][n] = A[m][k] * B[k][n];
        }
      }
    }
  }
  else if constexpr (FT == FType::rvv) {
  }
}

