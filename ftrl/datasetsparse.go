package ftrl

import (
	"log"
	"os"
	"text/tabwriter"
)

type Dataset struct {
	data      []Observation
	weightSum float64
	ncols     uint64
	nnz       uint64
}

//
func MakeDataset(capacity uint32) *Dataset {
	return &Dataset{
		data: make([]Observation, 0, capacity)}
}

// Add adds new element
func (d *Dataset) Add(o Observation) {
	d.data = append(d.data, o)

	// auxilary step
	d.weightSum += o.W
	d.nnz += uint64(len(o.X))
	for _, f := range o.X {
		k := uint64(f.Key)
		if k > d.ncols {
			d.ncols = k
		}
	}
}

// Row returns elements of ith row of dataset in sparse format
func (d *Dataset) Row(ith uint64) Observation {
	return d.data[ith]
}

// Label returns ith element of label vector
func (d *Dataset) Label(ith uint64) uint8 {
	return d.data[ith].Y
}

// SampleWeight returns ith element of sample weight vector
func (d *Dataset) SampleWeight(ith uint64) float64 {
	return d.data[ith].W
}

// WeightsSum return sum of weights of sample if dataset is weighted
// otherwise returns number of rows suppose each sample weight equals to 1
func (d *Dataset) WeightsSum() float64 {
	return d.weightSum
}

// Nnz returns numer of stored values
func (d *Dataset) Nnz() uint64 {
	return d.nnz
}

// NRows return number of stored rows
func (d *Dataset) NRows() uint64 {
	return uint64(len(d.data))
}

// NCols return number of stored cols
func (d *Dataset) NCols() uint64 {
	return d.ncols
}

func (d *Dataset) String() string {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	log.SetOutput(w)
	log.Println()
	log.Println("Field\tValue")
	log.Println("-----\t-----")
	log.Printf("rows\t%v", d.NRows())
	log.Printf("cols\t%v", d.NCols())
	log.Printf("nonzero\t%v", d.Nnz())
	w.Flush()

	return ""
}
