package ftrl

import (
	"log"
	"math"
	"os"
	"text/tabwriter"

	util "github.com/go-code/goFTRL/utils"
)

// FTRL is a structure for "Follow The Regularized Leader"
// logistic regression algorithm
type FTRL struct {
	weights    WeightKeeper
	params     Params
	activation LinkFunction
}

// MakeFTRL is fabric method for instance construction
func MakeFTRL(p Params) *FTRL {

	var f LinkFunction
	if p.activation == 'b' {
		f = util.SigmoidPiecewise
		// f = util.Sigmoid
	} else if p.activation == 'g' {
		f = util.Identity
	} else if p.activation == 'p' {
		f = util.Exp
	}

	return &FTRL{
		params:     p,
		activation: f,
		weights:    MakeWeightMap()}
}

// FitStream fits model from stream
func (a *FTRL) FitStream(stream DataStream, loss chan float64) {
	for o := range stream {
		x, y, sampleW := o.X, o.Y, o.W
		p := a.Predict(x)
		a.Update(x, p, y, sampleW)
		loss <- util.Logloss(p, y, sampleW)
	}
	close(loss)
}

// Fit fits model with given sample, label and sample weight
func (a *FTRL) Fit(o Observation) float64 {
	x, y, sampleW := o.X, o.Y, o.W
	p := a.Predict(x)
	a.Update(x, p, y, sampleW)
	return util.Logloss(p, y, sampleW)
}

// Predict return probability estimation of positive outcome
// for given sample
func (a *FTRL) Predict(s Sample) float64 {
	var p float64
	var w *weights
	var ok bool
	var k uint32
	var v float64
	for _, item := range s {
		k, v = item.Key, item.Value
		if w, ok = a.weights.Get(k); ok {
			p += w.get(a.params) * v
		}
	}
	return a.activation(p)
}

// Update updates weights of given sample features
func (a *FTRL) Update(s Sample, p float64, y uint8, sampleW float64) {

	g := util.Clip(sampleW*(p-float64(y)), a.params.clipgrad)

	var w *weights
	var ok bool
	var k uint32
	var v float64
	for _, item := range s {
		k, v = item.Key, item.Value

		if w, ok = a.weights.Get(k); !ok {
			w = &weights{}
			a.weights.Set(k, w)
		}

		zi, ni := w.zi, w.ni

		gi := g * v
		gi2 := gi * gi

		sigma := (math.Sqrt(ni+gi2) - math.Sqrt(ni)) / a.params.alpha
		wi := w.wi
		zi = zi + gi - sigma*wi
		ni = ni + gi2

		w.zi, w.ni = zi, ni
	}
}

// DecisionSummary prints learned weights summary
func (a *FTRL) DecisionSummary() {
	wcount := a.weights.Size()
	wnonzero, wmin, wmax := a.weights.Summary()

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	log.SetOutput(w)
	log.Println()
	log.Println("Decision summary\t:::::")
	log.Println("-----\t-----")
	log.Println(&a.params)
	log.Printf("weights count\t%v", wcount)
	log.Printf("count nonzero\t%v", wnonzero)
	log.Printf("min weight\t%v", wmin)
	log.Printf("max weight\t%v", wmax)
	w.Flush()
}
