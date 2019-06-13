package ftrl

import (
	"bufio"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

type SVMReader struct {
	fdata    string
	fweights string
	maxrows  uint32
}

type Pair struct {
	line   string
	weight string
}

func parseWorker(in <-chan *Pair, out DataStream, wg *sync.WaitGroup) {
	for pair := range in {
		line, weight := pair.line, pair.weight

		tokens := strings.Split(line, " ")
		y, _ := strconv.ParseUint(tokens[0], 10, 8)

		x := make(Sample, len(tokens[1:]))
		for i, token := range tokens[1:] {
			// [0] = key, [1] = value
			parts := strings.Split(token, ":")
			col, _ := strconv.ParseUint(parts[0], 10, 32)
			val, _ := strconv.ParseFloat(parts[1], 64)
			x[i] = Feature{Key: uint32(col), Value: val}
		}

		var w float64 = 1
		if weight != "" {
			w, _ = strconv.ParseFloat(weight, 64)
		}

		o := Observation{X: x, Y: uint8(y), W: w}
		out <- o
	}
	wg.Done()
}

func (r *SVMReader) Read(outstream DataStream) {

	data, err := os.Open(r.fdata)
	if err != nil {
		log.Fatal(err)
	}
	defer data.Close()
	scandata := bufio.NewScanner(data)

	var weights *os.File
	var scanwght *bufio.Scanner
	if r.fweights != "" {
		weights, err = os.Open(r.fweights)
		if err != nil {
			log.Fatal(err)
		}
		scanwght = bufio.NewScanner(weights)
	}
	defer weights.Close()

	// Pool of parsers
	var wg sync.WaitGroup
	raw := make(chan *Pair, 10000)
	for work := 0; work < runtime.NumCPU(); work++ {
		wg.Add(1)
		go parseWorker(raw, outstream, &wg)
	}

	// Forward file scan
	var row int
	var textX, textW string
	for scandata.Scan() {

		textX = scandata.Text()
		if scanwght.Scan() {
			textW = scanwght.Text()
		}
		raw <- &Pair{textX, textW}

		row++
		if r.maxrows == uint32(row) {
			break
		}

		if row%1000000 == 0 {
			log.Println(row)
		}
	}
	close(raw)

	go func() {
		wg.Wait()
		close(outstream)
	}()
}
