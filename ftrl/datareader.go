package ftrl

import (
	"runtime"
)

type DataReader struct {
	data     string
	weights  string
	colnames string

	cache   *DatasetSparse
	maxrows uint32
}

func NewDataReader(pdata, pweights, colnames string,
	prealloc, maxrows uint32) *DataReader {

	if pdata == "" || pweights == "" {
		return nil
	}

	dataset := MakeDataset(prealloc)

	return &DataReader{
		data:     pdata,
		weights:  pweights,
		colnames: colnames,
		cache:    dataset,
		maxrows:  maxrows,
	}
}

func (s *DataReader) Read() {
	r := SVMReader{
		fdata:    s.data,
		fweights: s.weights,
		maxrows:  s.maxrows,
	}

	ch := make(chan Observation, runtime.NumCPU())
	go r.Read(ch)
	for o := range ch {
		s.cache.Add(o)
	}
}

func (s *DataReader) GetData() *DatasetSparse {
	return s.cache
}
