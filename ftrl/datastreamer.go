package ftrl

import (
	"sync"
)

type Streamer struct {
	data     string
	weights  string
	colnames string

	usecache  bool
	cacheDone bool
	prealloc  uint32
	cache     *DatasetSparse

	maxrows uint32
}

func MakeStreamer(pdata, pweights, colnames string,
	usecache bool, prealloc, maxrows uint32) *Streamer {

	if pdata == "" || pweights == "" {
		return nil
	}

	var dataset *DatasetSparse
	if usecache {
		dataset = MakeDataset(prealloc)
	}

	return &Streamer{
		data:     pdata,
		weights:  pweights,
		colnames: colnames,

		usecache: usecache,
		prealloc: prealloc,
		cache:    dataset,
		maxrows:  maxrows,
	}
}

func (s *Streamer) ReadCache(out DataStream) {
	var wg sync.WaitGroup
	wg.Add(1)
	go func(w *sync.WaitGroup) {
		n := int(s.cache.NRows())
		for i := 0; i < n; i++ {
			out <- s.cache.Row(uint64(i))
		}
		wg.Done()
	}(&wg)

	go func() {
		wg.Wait()
		close(out)
	}()
}

func (s *Streamer) ReadAndCache(out DataStream, reader *SVMReader) {
	mirror := make(DataStream, 10000)
	go reader.Read(mirror)

	var wg sync.WaitGroup
	wg.Add(1)
	go func(w *sync.WaitGroup) {
		for o := range mirror {
			s.cache.Add(o)
			out <- o
		}
		wg.Done()
	}(&wg)

	go func() {
		wg.Wait()
		close(out)
		s.cacheDone = true
	}()
}

func (s *Streamer) Stream() DataStream {
	reader := SVMReader{
		fdata:    s.data,
		fweights: s.weights,
		maxrows:  s.maxrows,
	}

	out := make(DataStream, 10000)
	if s.usecache {
		if s.cacheDone {
			s.ReadCache(out)
		} else {
			s.ReadAndCache(out, &reader)
		}
		return out
	}

	go func() {
		reader.Read(out)
	}()

	return out
}
