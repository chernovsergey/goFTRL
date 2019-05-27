package ftrl

import (
	"log"
	"math"
	"runtime"
	"sync"

	ml "github.com/go-code/ml/utils"
)

const (
	DecisionOutputTemplate = "Weights count: %d. Nonzero: %d. Range: [%f, %f]"
	TrainOutputTemplate    = "#%d. tr.loss=%f grad.norm=%f"
	ValOutputTemplate      = "#%02d. tr.loss=%f val.loss=%f avg(pCTR)=%f grad.norm=%f"
)

// LinkFunction is an alias for activation function signature
type LinkFunction func(float64) float64

// FTRL is a structure for "Follow The Regularized Leader"
// logistic regression algorithm
type FTRL struct {
	weights    []*weights
	params     Params
	activation LinkFunction
}

// MakeFTRL is fabric method for instance construction
func MakeFTRL(p Params) *FTRL {

	// Choose activation function
	var f LinkFunction
	if p.activation == 'b' {
		// f = ml.SigmoidLinear
		f = ml.SigmoidPiecewise
		// f = ml.Sigmoid
	} else if p.activation == 'g' {
		f = ml.Identity
	} else if p.activation == 'p' {
		f = ml.Exp
	}

	return &FTRL{
		params:     p,
		activation: f,
		weights:    make([]*weights, 0)}
}

// Fit fits model for given dataset.
// Validation dataset enables overfitting detection
// mechanism, so final weights are chosen from best
// validation logloss
func (a *FTRL) Fit(train *ml.Dataset, valid *ml.Dataset) {
	numWeights := train.NCols()
	if valid != nil {
		if numWeights < valid.NCols() {
			numWeights = valid.NCols()
		}
	}
	a.initWeights(numWeights)

	var e uint64
	for e = 1; e <= a.params.niter; e++ {
		loss, gradnorm := epochRun(a, train)
		if valid != nil {
			lossVal, meanPred := a.Validate(valid)
			log.Printf(ValOutputTemplate, e, loss, lossVal, meanPred, gradnorm)
			continue
		}
		log.Printf(TrainOutputTemplate, e, loss, gradnorm)
	}
}

func (a *FTRL) initWeights(n uint64) {
	a.weights = make([]*weights, n)
}

// Predict return probability estimation of positive outcome
// for given sample
func (a *FTRL) Predict(s ml.Sample) float64 {
	var p float64
	var w *weights
	for _, feature := range s {
		k, v := feature.Key, feature.Value
		if a.weights[k] != nil {
			w = a.weights[k]
			p += w.get(a.params) * v
		}
	}

	return a.activation(p)
}

// PredictBatch return probability estimations for every
// sample in dataset
func (a *FTRL) PredictBatch(d *ml.Dataset) []float64 {
	nrows := d.NRows()
	nworkers := runtime.NumCPU()
	chunksize := int(nrows) / nworkers
	predicts := make([]float64, nrows)
	var wg sync.WaitGroup
	for i := 0; i < nworkers; i++ {
		start := i * chunksize
		end := start + chunksize
		if uint64(end) > nrows {
			end = int(nrows)
		}
		wg.Add(1)
		go predictBatchWorker(start, end, predicts, d, a, &wg)
	}
	wg.Wait()
	return predicts
}

func predictBatchWorker(start int, end int, arr []float64, d *ml.Dataset, a *FTRL, wg *sync.WaitGroup) {
	for j := start; j < end; j++ {
		idx := uint64(j)
		x := d.Row(idx)
		p := a.Predict(x)
		arr[j] = p
	}
	wg.Done()
}

// Save serializes model to file
func (a *FTRL) Save() {
	panic("Not implemented error")
}

// Load deserializes model from file
func (a *FTRL) Load() {
	panic("Not implemented error")
}

// ToJSON deserializes model weights to
// json format as input to any inference
// engine
func (a *FTRL) ToJSON() {
	panic("Not implemented error")
}

// GetWeights returns map index->weight for
// nonzero weights
func (a *FTRL) GetWeights() map[uint32]float64 {
	result := make(map[uint32]float64)
	for i, wptr := range a.weights {
		w := wptr.get(a.params)
		if w != 0 {
			result[uint32(i)] = w
		}
	}

	return result
}

// SetWeights assigns weights to model
func (a *FTRL) SetWeights() {
	panic("Not implemented error")
}

// GetParams returns model parameters
func (a *FTRL) GetParams() Params {
	return a.params
}

// SetParams assigns model parameters
func (a *FTRL) SetParams(p Params) {
	a.params = p
}

func processSample(a *FTRL, x ml.Sample, y uint8, w float64) (float64, float64) {
	p := a.Predict(x)
	gw := (p - float64(y))
	g := ml.Clip(w*gw, a.params.clipgrad)

	for _, feature := range x {
		k, v := feature.Key, feature.Value
		if a.weights[k] == nil {
			a.weights[k] = &weights{0.0, 0.0}
		}
		w := a.weights[k]

		zi, ni := w.zi, w.ni
		gi := g * v
		sigma := (math.Sqrt(ni+gi*gi) - math.Sqrt(ni)) / a.params.alpha
		wi := w.get(a.params)
		zi = zi + gi - sigma*wi
		ni = ni + (gi * gi)

		w.zi = zi
		w.ni = ni
	}

	return p, gw
}

func epochRun(a *FTRL, d *ml.Dataset) (float64, float64) {
	var i uint64
	nrows := d.NRows()
	grad := make([]float64, nrows)
	loss := 0.0
	for ; i < nrows; i++ {
		x := d.Row(i)
		y := d.Label(i)
		w := d.SampleWeight(i)
		p, g := processSample(a, x, y, w)

		grad[i] = g
		loss += ml.Logloss(p, y, w)
	}

	return loss / d.WeightsSum(), ml.Mean(grad)
}

// DecisionSummary prints summary about learned
// weights
func (a *FTRL) DecisionSummary() {
	numWeights := uint64(len(a.weights))
	var countNonzero int
	var minWeight, maxWeight float64
	var i uint64
	for i = 0; i < numWeights; i++ {
		if a.weights[i] == nil {
			continue
		}

		w := a.weights[i].get(a.params)
		if w != 0.0 {
			countNonzero++
		}

		minWeight = math.Min(minWeight, w)
		maxWeight = math.Max(maxWeight, w)
	}

	log.Printf(DecisionOutputTemplate,
		numWeights, countNonzero, minWeight, maxWeight)
}
