package ftrl

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

	return &Streamer{
		data:     pdata,
		weights:  pweights,
		colnames: colnames,
		usecache: usecache,
		prealloc: prealloc,
		maxrows:  maxrows,
	}
}

func (s *Streamer) streamFromCache(out DataStream) {
	n := int(s.cache.NRows())
	for i := 0; i < n; i++ {
		out <- s.cache.Row(uint64(i))
	}
}

func (s *Streamer) Stream(out DataStream) {
	if s.usecache {
		s.cache = MakeDataset(s.prealloc)
	}

	reader := SVMReader{
		fdata:    s.data,
		fweights: s.weights,
		maxrows:  s.maxrows,
	}

	if s.usecache {
		if s.cacheDone {
			go s.streamFromCache(out)
		} else {
			mirror := make(DataStream, 10000)
			go reader.Read(mirror)
			go func() {
				for o := range mirror {
					s.cache.Add(o)
					out <- o
				}
				close(out)
			}()
			go func() {
				s.cacheDone = true
			}()
		}
	} else {
		go reader.Read(out)
	}
}
