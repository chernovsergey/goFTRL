package ftrl

import (
	"log"
	"runtime"
	"sync"
	"time"

	util "github.com/go-code/goFTRL/utils"
)

const (
	TemplateTrain    = "#%02d. tr.loss=%f time{train:%v, val:%v}"
	TemplateTrainVal = "#%02d. tr.loss=%f val.loss=%f avg(pCTR)=%f time{train:%v, val:%v}"
)

type Trainer struct {
	model     *FTRL
	train     *Streamer
	valid     *Streamer
	iters     uint32
	optimized bool
}

type result struct {
	loss  float64
	wsum  float64
	psum  float64
	iters uint64
}

func MakeTrainer(model *FTRL, trainStream *Streamer, valStream *Streamer,
	numEpoch uint32) *Trainer {
	return &Trainer{
		model: model,
		train: trainStream,
		valid: valStream,
		iters: numEpoch,
	}
}

func (t *Trainer) optimizeStorage() {
	switch t.model.weights.(type) {
	case *WeightMap:
		ts := time.Now()
		t.model.weights = MakeWeightArray(t.model.weights.(*WeightMap))
		t.optimized = true
		log.Println("weights store = array", time.Since(ts))
	}
}

func (t *Trainer) Run() {
	t0 := time.Now()
	for i := 0; i < int(t.iters); i++ {

		if t.train.cacheDone && !t.optimized {
			t.optimizeStorage()
		}

		tic := time.Now()
		res := t.Train()
		t1 := time.Since(tic)
		loss := res.loss / res.wsum

		if t.valid != nil {
			tic = time.Now()
			res = t.Validate()
			t2 := time.Since(tic)

			valloss := res.loss / res.wsum
			avgPred := res.psum / res.wsum
			log.Printf(TemplateTrainVal, i+1, loss, valloss, avgPred, t1, t2)
			continue
		}
		log.Printf(TemplateTrain, i+1, loss, t1)
	}
	log.Println("Total fit time:", time.Since(t0))
}

func (t *Trainer) TrainParallel(input DataStream) result {

	loss := make(chan result, runtime.NumCPU())

	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func(w *sync.WaitGroup, out chan result) {
			var res result
			for o := range input {
				ll := t.model.Fit(o)
				res.loss += ll
			}
			out <- res
			w.Done()
		}(&wg, loss)
	}

	go func() {
		wg.Wait()
		close(loss)
	}()

	var res result
	for ll := range loss {
		res.loss += ll.loss
		res.wsum++
	}

	return res
}

func (t *Trainer) Train() result {
	input := t.train.Stream()

	if t.train.cacheDone {
		return t.TrainParallel(input)
	}

	loss := make(chan float64)
	go func() {
		t.model.FitStream(input, loss)
	}()

	var res result
	for ll := range loss {
		res.loss += ll
		res.wsum++
	}

	return res
}

func (t *Trainer) Validate() result {

	input := t.valid.Stream()

	results := make(chan result, runtime.NumCPU())

	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func(w *sync.WaitGroup, out chan result) {
			var res result
			for o := range input {
				p := t.model.Predict(o.X)
				res.loss += util.Logloss(p, o.Y, o.W)
				res.iters++
				res.wsum += o.W
				res.psum += p * o.W
			}
			out <- res
			w.Done()
		}(&wg, results)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var res result
	for r := range results {
		res.loss += r.loss
		res.wsum += r.wsum
		res.psum += r.psum
		res.iters += r.iters
	}

	return res
}

func (t *Trainer) PrintSummary() {
	t.model.DecisionSummary()
	log.Println(t.train.cache)
	log.Println(t.valid.cache)
}
