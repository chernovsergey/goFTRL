package utils

import "math/rand"

func max(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

// COOMatrix is for storing coordinates
// Use this type for CSR matrix construction
type COOMatrix struct {
	row []uint64
	col []uint64
	dat []float64

	nnz      uint64
	n        uint64
	m        uint64
	isBinary bool

	nrows uint64
	ncols uint64
}

// FromArrayTriplet constucts COO matrix
// from triplet of arrays: rows, cols, data
func (mat *COOMatrix) FromArrayTriplet(r, c []uint64, d []float64) {
	mat.row = r
	mat.col = c
	mat.dat = d
}

// Set sets new value at particular coordinate
func (mat *COOMatrix) Set(i, j uint64, v float64) {
	if v == 0 {
		return
	}

	mat.row = append(mat.row, i)
	mat.col = append(mat.col, j)
	if !mat.isBinary {
		mat.dat = append(mat.dat, v)
	}

	mat.n = max(mat.n, i)
	mat.m = max(mat.m, j)
	mat.nnz++

	mat.nrows = mat.n + 1
	mat.ncols = mat.m + 1
}

// GetShape returns shape of yet constructed matrix
func (mat *COOMatrix) GetShape() map[string]uint64 {
	return map[string]uint64{"rows": mat.n + 1, "cols": mat.m + 1}
}

// ShuffleRows shuffles rows of matrix
func (mat *COOMatrix) ShuffleRows() {
	rand.Seed(42) // time.Now().UnixNano()
	rand.Shuffle(len(mat.row)-1, func(i, j int) {
		mat.row[i], mat.row[j] = mat.row[j], mat.row[i]
		mat.col[i], mat.col[j] = mat.col[j], mat.col[i]
		if !mat.isBinary {
			mat.dat[i], mat.dat[j] = mat.dat[j], mat.dat[i]
		}
	})
}

// MakeCOO creates empty COO matrix
func MakeCOO(binary bool) *COOMatrix {
	return &COOMatrix{isBinary: binary}
}
