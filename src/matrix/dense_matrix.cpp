#include "dense_matrix.h"

#include <assert.h>

DenseMatrix::DenseMatrix(uint32_t rows, uint32_t cols)
    : rows{rows}, cols{cols}, matrix((size_t)rows * cols, 0.0)
{
}

void DenseMatrix::mvp(const double *__restrict x, double *__restrict y) const
{
	for (uint32_t i = 0; i < rows; i++) {
		double acc = 0.0;
		const double *row = &matrix.data[(size_t)i * cols];
		for (uint32_t j = 0; j < cols; j++)
			acc += row[j] * x[j];
		y[i] = acc;
	}
}

double DenseMatrix::sum() const
{
	double acc = 0.0;
	for (size_t k = 0; k < matrix.size; k++)
		acc += matrix.data[k];
	return acc;
}

double &DenseMatrix::operator()(uint32_t i, uint32_t j)
{
	return at(i, j);
}

double &DenseMatrix::at(uint32_t i, uint32_t j)
{
	assert(i < rows && j < cols);
	return matrix.data[(size_t)i * cols + j];
}

double DenseMatrix::at(uint32_t i, uint32_t j) const
{
	assert(i < rows && j < cols);
	return matrix.data[(size_t)i * cols + j];
}
