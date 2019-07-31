package ftrl

import (
	"fmt"
	"log"
	"math"
	"os"
	"text/tabwriter"

	util "github.com/go-code/goFTRL/utils"
)

type Params struct {
	alpha, beta, lambda1, lambda2 float64
	clipgrad                      float64
	dropout                       float64
	tol                           float64
	niter                         uint64
	activation                    rune
}

func MakeParams(
	a, b, l1, l2, clipgrad, dropout, tol float64,
	maxiter uint64, activation rune) Params {
	return Params{
		alpha:      a,
		beta:       b,
		lambda1:    l1,
		lambda2:    l2,
		clipgrad:   clipgrad,
		dropout:    dropout,
		tol:        tol,
		niter:      maxiter,
		activation: activation}
}

func (p *Params) String() string {
	return fmt.Sprintf("FTRL{Alpha:%v, Beta:%v, L1:%v, L2:%v, max_iter:%v, activation:%q}",
		p.alpha, p.beta, p.lambda1, p.lambda2, p.niter, p.activation)
}

// FTRL is a structure for "Follow The Regularized Leader"
// logistic regression algorithm
type FTRL struct {
	weights    []*weights
	params     Params
	activation LinkFunction
}

// MakeFTRL is fabric method for instance construction
func MakeFTRL(p Params, warmstart string) *FTRL {

	var f LinkFunction
	if p.activation == 'b' {
		f = util.Sigmoid
	} else if p.activation == 'g' {
		f = util.Identity
	} else if p.activation == 'p' {
		f = util.Exp
	}

	return &FTRL{
		params:     p,
		activation: f,
		weights:    make([]*weights, 0, 0),
	}
}

func (a *FTRL) AllocWeightsStore(size uint64) {
	if len(a.weights) != 0 {
		panic("Ooops! Weights are already allocated")
	}
	a.weights = make([]*weights, size, size)
}

// Fit fits model with given sample, label and sample weight
func (a *FTRL) Fit(o Observation) float64 {
	x, y, sampleW := o.X, o.Y, o.W
	p := a.Predict(x)
	a.Update(x, p, y, sampleW)
	return util.Logloss(p, y, sampleW)
}

// Predict returns probability estimation of positive outcome
// for given sample
func (a *FTRL) Predict(s Sample) float64 {
	var p float64
	var w *weights
	var k uint32
	var v float64

	numWeights := uint32(len(a.weights))
	for _, item := range s {
		k, v = item.Key, item.Value

		if k >= numWeights {
			continue
		}
		w = a.weights[k]
		if w != nil {
			p += w.get(a.params) * v
		}
	}
	return a.activation(p)
}

// EstimateUCB estimates upper confidence bound of prediction
// for given sample
func (a *FTRL) EstimateUCB(s Sample) float64 {
	var ub float64
	numWeights := uint32(len(a.weights))
	for _, item := range s {
		k, v := item.Key, item.Value
		if k >= numWeights {
			continue
		}
		w := a.weights[k]
		if w != nil {
			ub += v / math.Sqrt(w.ni)
		}
	}

	return a.params.alpha * ub
}

// Update updates weights of given sample features
func (a *FTRL) Update(s Sample, p float64, y uint8, sampleW float64) {

	g := util.Clip(sampleW*(p-float64(y)), a.params.clipgrad)

	var w *weights
	var k uint32
	var v float64
	for _, item := range s {
		k, v = item.Key, item.Value

		w = a.weights[k]
		if w == nil {
			w = &weights{}
			a.weights[k] = w
		}

		zi, ni := w.zi, w.ni

		gi := g * v
		gi2 := gi * gi

		sigma := (math.Sqrt(ni+gi2) - math.Sqrt(ni)) / a.params.alpha
		wi := w.wi
		zi = zi + gi - sigma*wi
		ni = ni + gi2

		w.set(zi, ni)
	}
}

func (a *FTRL) Copy() FTRL {
	w := make([]*weights, len(a.weights), cap(a.weights))
	for k, v := range a.weights {
		newW := *v
		w[k] = &newW
	}
	cp := FTRL{
		weights:    w,
		params:     a.params,
		activation: a.activation,
	}

	return cp
}

func (a *FTRL) Save(path string) {

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	for idx, w := range a.weights {
		if w == nil {
			f.WriteString(fmt.Sprintf("%d:%f\n", idx, float64(0.0)))
			continue
		}
		f.WriteString(fmt.Sprintf("%d:%f\n", idx, w.wi))
	}
	log.Println("Saved model to file:", path)
}

// DecisionSummary prints learned weights summary
func (a *FTRL) DecisionSummary() {

	var nnzCount uint32
	var min, max float64
	for _, w := range a.weights {
		if w == nil {
			continue
		}
		if w.wi != 0.0 {
			nnzCount++
		}
		min = math.Min(min, w.wi)
		max = math.Max(max, w.wi)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	log.SetOutput(w)
	log.Println()
	log.Println("Decision summary\t:::::")
	log.Println("-----\t-----")
	log.Println(&a.params)
	log.Printf("weights count\t%v", len(a.weights))
	log.Printf("count nonzero\t%v", nnzCount)
	log.Printf("min weight\t%v", min)
	log.Printf("max weight\t%v", max)
	w.Flush()
}
