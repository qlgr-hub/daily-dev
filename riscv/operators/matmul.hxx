#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <riscv_vector.h>


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
  T* get() {
    return __data.get();
  }

  const T* get() const {
    return __data.get();
  }

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
    const T* ptrA = A.get();
    const T* ptrB = B.get();
    T* ptrC = C.get();

    // Assuming row-major storage.
    // This implementation vectorizes the inner loop (over K) using RVV reductions.
    // Basic broadcasting is supported naturally when dimensions align (e.g., if K=1, it acts like outer product/broadcast).
    for (uint64_t m = 0; m < A.rows(); ++m) {
      for (uint64_t n = 0; n < B.cols(); ++n) {
        T sum = T(0);
        uint64_t k = 0;

        // Main loop using larger LMUL (m8) for efficiency if possible.
        uint64_t vlmax_m8 = __riscv_vsetvlmax_e32m8();
        for (; k + vlmax_m8 <= A.cols(); k += vlmax_m8) {
          size_t vl = __riscv_vsetvl_e32m8(A.cols() - k);
          vfloat32m8_t vec_a = __riscv_vle32_v_f32m8(&ptrA[m * A.cols() + k], vl);
          vfloat32m8_t vec_b = __riscv_vlse32_v_f32m8(&ptrB[k * B.cols() + n], B.cols() * sizeof(float), vl);  // Strided load for B
          vfloat32m8_t mul = __riscv_vfmul_vv_f32m8(vec_a, vec_b, vl);
          // Reduce to scalar (unordered sum reduction)
          vfloat32m1_t red = __riscv_vfredusum_vs_f32m8_f32m1(mul, __riscv_vfmv_s_f_f32m1(__riscv_vundefined_f32m1(), 0.0f), vl);
          sum += __riscv_vfmv_f_s_f32m1_f32(red);
        }

        // Tail loop with m1.
        uint64_t vl = __riscv_vsetvl_e32m1(A.cols() - k);
        if (vl > 0) {
          vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&A[m * A.cols() + k], vl);
          vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1(&B[k * B.cols() + n], B.cols() * sizeof(float), vl);
          vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
          vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(mul, __riscv_vfmv_s_f_f32m1(__riscv_vundefined_f32m1(), 0.0f), vl);
          sum += __riscv_vfmv_f_s_f32m1_f32(red);
        }

        ptrC[m * B.cols() + n] = sum;
      }
    }
  }
}

