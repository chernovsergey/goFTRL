package ml

import (
	"fmt"
	"reflect"
	"testing"
)

func TestCOOBuild(t *testing.T) {
	r := []uint32{0, 0, 1, 2, 2, 2}
	c := []uint32{0, 2, 2, 0, 1, 2}
	d := []float32{1, 2, 3, 4, 5, 6}

	N := len(r)

	mat := MakeCOO(false)
	for i := 0; i < N; i++ {
		mat.Set(r[i], c[i], d[i])
	}

	if mat.nrows != 3 {
		t.Error("nrows != 3")
	}

	if mat.ncols != 3 {
		t.Error("ncols != 3")
	}

	if mat.nnz != 6 {
		t.Error("nnz != 6")
	}
	fmt.Println(mat)
}

func TestCSRBuildFromCOO(t *testing.T) {
	r := []uint32{0, 0, 1, 2, 2, 2}
	c := []uint32{0, 2, 2, 0, 1, 2}
	d := []float32{1, 2, 3, 4, 5, 6}

	N := len(r)

	coo := MakeCOO(false)
	for i := 0; i < N; i++ {
		coo.Set(r[i], c[i], d[i])
	}

	csr := MakeCSR(false)
	csr.FromCOO(coo)
	fmt.Println(csr)

	if !reflect.DeepEqual(csr.ia, []uint32{0, 2, 3, 6}) {
		t.Error("IA wrong")
	}

	if !reflect.DeepEqual(csr.ja, c) {
		t.Error("JA wrong")
	}

	if !reflect.DeepEqual(csr.dat, d) {
		t.Error("DAT wrong")
	}
}

func TestCSRGetRowMethod(t *testing.T) {
	r := []uint32{0, 0, 1, 2, 2, 2}
	c := []uint32{0, 2, 2, 0, 1, 2}
	d := []float32{1, 2, 3, 4, 5, 6}

	N := len(r)

	coo := MakeCOO(false)
	for i := 0; i < N; i++ {
		coo.Set(r[i], c[i], d[i])
	}

	csr := MakeCSR(false)
	csr.FromCOO(coo)

	one := csr.GetRow(0)
	two := csr.GetRow(1)
	three := csr.GetRow(2)

	if !reflect.DeepEqual(one, map[uint32]float32{0: 1, 2: 2}) {
		t.Error("Wrong GetRow(0)")
		fmt.Println(one)
	}

	if !reflect.DeepEqual(two, map[uint32]float32{2: 3}) {
		t.Error("Wrong GetRow(1)")
		fmt.Println(two)
	}

	if !reflect.DeepEqual(three, map[uint32]float32{0: 4, 1: 5, 2: 6}) {
		t.Error("Wrong GetRow(2)")
		fmt.Println(three)
	}
}

func TestCSRBinaryGetRowMethod(t *testing.T) {
	r := []uint32{0, 0, 1, 2, 2, 2}
	c := []uint32{0, 2, 2, 0, 1, 2}
	d := []float32{1, 2, 3, 4, 5, 6}

	N := len(r)

	coo := MakeCOO(true)
	for i := 0; i < N; i++ {
		coo.Set(r[i], c[i], d[i])
	}

	csr := MakeCSR(true)
	csr.FromCOO(coo)

	one := csr.GetRow(0)
	two := csr.GetRow(1)
	three := csr.GetRow(2)

	if !reflect.DeepEqual(one, map[uint32]float32{0: 1, 2: 1}) {
		t.Error("Wrong GetRow(0)")
		fmt.Println(one)
	}

	if !reflect.DeepEqual(two, map[uint32]float32{2: 1}) {
		t.Error("Wrong GetRow(1)")
		fmt.Println(two)
	}

	if !reflect.DeepEqual(three, map[uint32]float32{0: 1, 1: 1, 2: 1}) {
		t.Error("Wrong GetRow(2)")
		fmt.Println(three)
	}
}
