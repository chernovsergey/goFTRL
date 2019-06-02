package utils

import (
	"bufio"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

type Feature struct {
	Key   uint64
	Value float64
}

type Sample []Feature

// Dataset currently implements basic manipulations with data
// such as loading from file, row iteration and
// providing meta information
type Dataset struct {
	data          *CSRMatrix
	labels        []uint8
	isWeighted    bool
	meanTarget    float64
	weightsSum    float64
	sampleWeights []float64
	featureNames  []string
}

// Shape returns tuple with sizes for each dimension
func (d *Dataset) Shape() (uint64, uint64) {
	return d.data.nrows, d.data.ncols
}

// Sparcity computes ratio of nonzero elements to all elements
func (d *Dataset) Sparcity() float64 {
	return float64(d.data.nnz) / float64(d.data.nrows*d.data.ncols)
}

// Row returns elements of ith row of dataset in sparse format
func (d *Dataset) Row(ith uint64) Sample {
	return d.data.GetRow(ith)
}

// Label returns ith element of label vector
func (d *Dataset) Label(ith uint64) uint8 {
	return d.labels[ith]
}

// SampleWeight returns ith element of sample weight vector
func (d *Dataset) SampleWeight(ith uint64) float64 {
	if !d.isWeighted {
		return 1.0
	}
	return d.sampleWeights[ith]
}

// WeightsSum return sum of weights of sample if dataset is weighted
// otherwise returns number of rows suppose each sample weight equals to 1
func (d *Dataset) WeightsSum() float64 {
	if !d.isWeighted {
		return float64(d.NRows())
	}
	return d.weightsSum
}

// NameOfCol returns name of ith column
func (d *Dataset) NameOfCol(ith uint64) string {
	return d.featureNames[ith]
}

// Nnz returns numer of stored values
func (d *Dataset) Nnz() uint64 {
	return d.data.nnz
}

// NRows return number of stored rows
func (d *Dataset) NRows() uint64 {
	return d.data.nrows
}

// NCols return number of stored cols
func (d *Dataset) NCols() uint64 {
	return d.data.ncols
}

// MeanTarget return average probability
// of outcome for dataset
func (d *Dataset) MeanTarget() float64 {
	return d.meanTarget
}

// ReadSVM parses input file in libsvm format
// simutaniously updating COO matrix. Finally compresses
// COO matrix to CSR format.
func (d *Dataset) ReadSVM(path string,
	maxrows int32, isBinary bool) (*COOMatrix, []uint8) {

	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	matrix := MakeCOO(isBinary)
	labels := make([]uint8, 0)
	var rowIdx uint64
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}

		line = strings.TrimSpace(line)
		tokens := strings.Split(line, " ")

		label, err := strconv.ParseUint(tokens[0], 10, 8)
		if err != nil {
			log.Fatal(err)
		}
		labels = append(labels, uint8(label))

		for _, token := range tokens[1:] {
			// [0] = key, [1] = value
			parts := strings.Split(token, ":")
			colIdx, _ := strconv.ParseUint(parts[0], 10, 32)
			if isBinary {
				matrix.Set(rowIdx, uint64(colIdx), 1)
			} else {
				val, _ := strconv.ParseFloat(parts[1], 64)
				matrix.Set(rowIdx, uint64(colIdx), val)
			}
		}

		rowIdx++
		if maxrows == int32(rowIdx) {
			break
		}
	}
	return matrix, labels
}

func (d *Dataset) LoadSampleWeights(path string) ([]float64, float64, error) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
		return nil, 0, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	weights := make([]float64, 0)
	wsum := 0.0
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}

		line = strings.TrimSpace(line)
		w, err := strconv.ParseFloat(line, 64)
		if err != nil {
			log.Fatal(err)
		}

		weights = append(weights, w)
		wsum += w
	}
	return weights, wsum, nil
}

func (d *Dataset) LoadFeatureNames(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	names := make([]string, 0)
	var idx uint64
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}

		line = strings.TrimSpace(line)
		names = append(names, line)

		idx++
	}
	return names, nil
}

// FromCSVFile reads file to dataset via
// rows --> coo matrix --> csr matrix transformation
func (d *Dataset) FromCSVFile(path string, maxrows int32) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// TODO
	// parse csv rows
	// zip values with index columns
	// add to coo
	// make csr compression
}

// MakeDataset creates Dataset object
func MakeDataset() *Dataset {
	return &Dataset{}
}

func LoadDataset(path, pathW, pathF string,
	nrows int32, binary bool, shuffle bool) *Dataset {

	d := MakeDataset()
	coo, labels := d.ReadSVM(path, nrows, binary)
	csr := MakeCSR(binary)
	csr.FromCOO(coo)
	csr.CacheRows()
	d.data = csr
	d.labels = labels

	var err error

	var weights []float64
	var wsum float64
	if pathW != "" {
		weights, wsum, err = d.LoadSampleWeights(pathW)
		if err != nil {
			panic("Can't load sample weights")
		}
		d.isWeighted = true
		d.sampleWeights = weights
		d.weightsSum = wsum
	}

	var fnames []string
	if pathF != "" {
		fnames, err = d.LoadFeatureNames(pathF)
		if err != nil {
			panic("Can't load feature names")
		}
		d.featureNames = fnames
	}

	if shuffle {
		log.Println("Shuffling dataset ...")
		m, n, k := len(d.data.cache), len(d.sampleWeights), len(d.labels)
		if m != n || m != k {
			panic("Inconsistent shape of data/weights/labels")
		}

		rand.Seed(time.Now().UnixNano()) // or 42
		rand.Shuffle(len(d.data.cache), func(i, j int) {
			d.data.cache[i], d.data.cache[j] = d.data.cache[j], d.data.cache[i]
			d.labels[i], d.labels[j] = d.labels[j], d.labels[i]
			d.sampleWeights[i], d.sampleWeights[j] = d.sampleWeights[j], d.sampleWeights[i]
		})
	}

	log.Println(d)
	return d
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
	log.Printf("sparcity\t%v", d.Sparcity())
	log.Printf("binary\t%v", d.data.isBinary)
	log.Printf("weighed\t%v", d.isWeighted)
	w.Flush()

	return ""
}
