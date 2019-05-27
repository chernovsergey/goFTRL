package ml

import (
	"testing"
)

// BenchmarkCSRFromCOO benchmarks csr construction performance
func BenchmarkCSRFromCOO(b *testing.B) {
	r := []uint32{0, 0, 1, 2, 2, 2}
	c := []uint32{0, 2, 2, 0, 1, 2}
	d := []float32{1, 2, 3, 4, 5, 6}

	N := len(r)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {

		coo := MakeCOO(false)
		for i := 0; i < N; i++ {
			coo.Set(r[i], c[i], d[i])
		}
		b.StartTimer()
		csr := MakeCSR(false)
		csr.FromCOO(coo)
		b.StopTimer()
	}

}

// BenchmarkCSRGetRow benchmarks row access performance
func BenchmarkCSRGetRow(b *testing.B) {
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

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		csr.GetRow(1)
	}
}

func BenchmarkDatasetGetRow(b *testing.B) {
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

	dataset := Dataset{data: csr}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dataset.Row(1)
	}
}
