package ml

import "runtime"

// CSRMatrix is compressed sparse row matrix
type CSRMatrix struct {
	dat []float64
	ia  []uint64
	ja  []uint64

	n        uint64
	m        uint64
	nnz      uint64
	isBinary bool

	nrows uint64
	ncols uint64

	isCached bool
	cache    []Sample
}

// FromCOO builds sparse row matrix through
// iteration over COO matrix
func (csr *CSRMatrix) FromCOO(coo *COOMatrix) {

	csr.ia = make([]uint64, coo.nrows+1)
	csr.ja = make([]uint64, coo.nnz)
	if !csr.isBinary {
		csr.dat = make([]float64, coo.nnz)
	}

	// compute number of non-zero entries per row
	var i uint64
	for i = 0; i < coo.nnz; i++ {
		csr.ia[coo.row[i]]++
	}

	// cumsum the nnz per row to get csr.ia
	// ia[0] = 0
	// ia[1] = ia[0] + count nnz elements at row 1
	// ia[2] = ia[1] + count nnz elements at row 2
	// etc.
	var cumsum uint64
	for i = 0; i < coo.nrows; i++ {
		temp := csr.ia[i]
		csr.ia[i] = cumsum
		cumsum += temp
	}
	csr.ia[coo.nrows] = cumsum

	// write coo.col, coo.dat into csr.ja, csr.dat
	for i = 0; i < coo.nnz; i++ {
		row := coo.row[i]
		dest := csr.ia[row]

		csr.ja[dest] = coo.col[i]
		if !csr.isBinary {
			csr.dat[dest] = coo.dat[i]
		}

		csr.ia[row]++
	}

	var last uint64
	for i = 0; i <= coo.n; i++ {
		temp := csr.ia[i]
		csr.ia[i] = last
		last = temp
	}

	// copy metadata
	csr.n = coo.n
	csr.m = coo.m
	csr.nnz = coo.nnz

	csr.nrows = coo.nrows
	csr.ncols = coo.ncols
}

// BuildRow constructs map from i-th row data
// like this: {col1: val1, col2:val2, ...}
// where col's are the column indexes and
// val's are the values at that columns
func (csr *CSRMatrix) BuildRow(ith uint64) Sample {
	l := csr.ia[ith]
	r := csr.ia[ith+1]
	cols := csr.ja[l:r]
	var vals []float64
	if !csr.isBinary {
		vals = csr.dat[l:r]
	}

	size := len(cols)
	result := make(Sample, size)
	for i := 0; i < size; i++ {
		if !csr.isBinary {
			result[i] = Feature{cols[i], vals[i]}
		} else {
			result[i] = Feature{cols[i], 1.0}
		}
	}

	return result
}

func (csr *CSRMatrix) CacheRows() {
	cache := make([]Sample, csr.nrows)
	nworkers := runtime.NumCPU()
	chunksize := int(csr.nrows) / nworkers
	for i := 0; i < nworkers; i++ {
		start := i * chunksize
		end := start + chunksize
		go func() {
			for j := start; j < end; j++ {
				cache[j] = csr.BuildRow(uint64(j))
			}
		}()
	}
	csr.isCached = true
	csr.cache = cache
}

func (csr *CSRMatrix) GetRow(ith uint64) Sample {
	if !csr.isCached {
		return csr.BuildRow(ith)
	}
	return csr.cache[ith]
}

// MakeCSR creates empty csr matrix
func MakeCSR(binary bool) *CSRMatrix {
	return &CSRMatrix{isBinary: binary}
}
