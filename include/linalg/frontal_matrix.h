#ifndef FRONTAL_MATRIX_H
#define FRONTAL_MATRIX_H

#include <assert.h>
#include <stdint.h>

/******************************************************************************
 *
 * SymFrontMatrix : the dense symmetric frontal matrix of one multifrontal
 * step, as a NON OWNING view on a caller provided buffer.
 *
 * Storage is column major and UNPACKED : a front of order m occupies m * m
 * doubles, but only the lower triangle (i >= j) is ever read or written, so
 * the strictly upper triangle holds garbage at all times. Packed storage
 * (m(m+1)/2) would halve the footprint, but its BLAS (dspr) is unblocked and
 * it is incompatible with the leading dimension that dpotrf / dsyrk need,
 * which is what a future supernodal version would be built on. Unpacked
 * costs one extra buffer's worth of memory -- the scratch is allocated once,
 * at the maximum front order -- and keeps indexing trivial.
 *
 *****************************************************************************/
struct SymFrontMatrix {
	uint32_t m;      /* order of the front */
	double  *data;   /* m * m doubles, column major, lower triangle only */

	SymFrontMatrix(uint32_t m, double *data) : m{m}, data{data} {}

	/* Only the lower triangle exists : i >= j is a hard invariant, not a
	 * convenience, since the upper half is never maintained. */
	double &at(uint32_t i, uint32_t j)
	{
		assert(j <= i && i < m);
		return data[(size_t)j * m + i];
	}
	double at(uint32_t i, uint32_t j) const
	{
		assert(j <= i && i < m);
		return data[(size_t)j * m + i];
	}
	/* First stored element of column j, i.e. the diagonal entry (j, j). */
	double *col(uint32_t j)
	{
		assert(j < m);
		return &data[(size_t)j * m + j];
	}

	void zeroLower()
	{
		for (uint32_t j = 0; j < m; j++) {
			double *c = col(j);
			for (uint32_t i = 0; i < m - j; i++)
				c[i] = 0.0;
		}
	}
};

#endif // FRONTAL_MATRIX_H
