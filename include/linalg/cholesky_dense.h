#ifndef CHOLESKY_DENSE_H
#define CHOLESKY_DENSE_H

#include "array.h"

#include "dense_matrix.h"

#include <stdint.h>
#include <memory>

/// @brief Does exactly one step of the dense Cholesky factorization in the recursive format, 
///        where the recursion is performed on the Schur complement.
/// @details RL = RightLooking
/// @param L dense Cholesky factor: A=LL^T. It's a vector represented as a DenseMatrix
/// @param S Schur complement at step k
/// @param k number of the current step
void schurCholeskyOneStep(DenseMatrix* L, DenseMatrix* S, uint32_t k);

/// @brief Does exactly one step of the dense Cholesky factorization in the recursive format, 
///        where the recursion is performed on the Schur complement. 
/// @details RL = RightLooking
/// @param L dense Cholesky factor: A=LL^T. It's a vector represented as a TArray<double>
/// @param S Schur complement at step k
/// @param k number of the current step
void schurCholeskyOneStep(std::shared_ptr<TArray<double>> L, DenseMatrix* S, uint32_t k);

#endif // CHOLESKY_DENSE_H
