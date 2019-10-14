package ftrl

import (
	"bufio"
	"log"
	"os"
	"runtime"
)

type DataReader struct {
	data     string
	weights  string
	colnames string

	cache   *Dataset
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

	f, err := os.Open(s.colnames)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		s.cache.featurenames = append(s.cache.featurenames, scanner.Text())
	}
}

func (s *DataReader) GetData() *Dataset {
	return s.cache
}
