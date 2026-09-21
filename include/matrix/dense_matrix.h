#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include "array.h"
#include <stdint.h>

/* Plain row major dense matrix. Every entry is stored : the symmetric
 * frontal matrices of the multifrontal factorization do NOT use this type,
 * they use SymFrontMatrix (frontal_matrix.h), which only ever touches its
 * lower triangle and would therefore make sum() and mvp() below wrong. */
struct DenseMatrix {
	protected:
	 	uint32_t rows, cols;
		TArray<double> matrix;

	public:
	 	DenseMatrix(uint32_t rows, uint32_t cols);
		uint32_t nRows() const { return rows; }
		uint32_t nCols() const { return cols; }
		/* Matrix vector product : Ax -> y */
		virtual void mvp(const double *__restrict x, double *__restrict y) const;
		/* Sum of matrix elements */
		virtual double sum() const;
		virtual ~DenseMatrix() = default;
		double &operator()(uint32_t i, uint32_t j);
		double &at(uint32_t i, uint32_t j);
		double at(uint32_t i, uint32_t j) const;
};

#endif // DENSE_MATRIX_H
