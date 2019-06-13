package ftrl

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type SVMReader struct {
	fdata    string
	fweights string
	maxrows  uint32
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

	var row int
	for scandata.Scan() {

		sampleRow := strings.Split(strings.TrimSpace(scandata.Text()), " ")

		y, err := strconv.ParseUint(sampleRow[0], 10, 8)
		if err != nil {
			log.Fatal(err)
		}

		x := make(Sample, len(sampleRow[1:]))
		for i, token := range sampleRow[1:] {
			// [0] = key, [1] = value
			parts := strings.Split(token, ":")
			col, _ := strconv.ParseUint(parts[0], 10, 32)
			val, _ := strconv.ParseFloat(parts[1], 64)
			x[i] = Feature{Key: uint32(col), Value: val}
		}

		var w float64 = 1
		if scanwght.Scan() {
			weightRow := strings.TrimSpace(scanwght.Text())
			w, _ = strconv.ParseFloat(weightRow, 64)
		}

		o := Observation{X: x, Y: uint8(y), W: w}
		outstream <- o

		row++
		if r.maxrows == uint32(row) {
			break
		}

		if row%100000 == 0 {
			log.Println(row)
		}
	}

	close(outstream)
}
