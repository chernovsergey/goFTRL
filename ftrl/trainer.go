package ftrl

import (
	"log"
	"runtime"
	"sync"

	util "github.com/go-code/goFTRL/utils"
)

const (
	TemplateTrain    = "#%02d. tr.loss=%f"
	TemplateTrainVal = TemplateTrain + " val.loss=%f avg(pCTR)=%f"
)

type Trainer struct {
	model     *FTRL
	streamer  *Streamer
	valstream *Streamer
	iters     uint32
}

type result struct {
	loss  float64
	wsum  float64
	psum  float64
	iters uint64
}

func MakeTrainer(model *FTRL, trainStream *Streamer, valStream *Streamer, numEpoch uint32) *Trainer {
	return &Trainer{
		model:     model,
		streamer:  trainStream,
		valstream: valStream,
		iters:     numEpoch,
	}
}

func (t *Trainer) Run() {
	for i := 0; i < int(t.iters); i++ {
		tr := t.Train()
		trloss := tr.loss / tr.wsum
		if t.valstream != nil {
			val := t.Validate()
			log.Printf(TemplateTrainVal, i+1, trloss,
				val.loss/val.wsum, val.psum/float64(val.iters))
			continue
		}
		log.Printf(TemplateTrain, i+1, trloss)
	}
}

func (t *Trainer) Train() result {
	input := t.streamer.Stream()

	loss := make(chan float64)
	go func() {
		t.model.Fit(input, loss)
	}()

	var res result
	for ll := range loss {
		res.loss += ll
	}
	res.wsum = t.streamer.cache.WeightsSum()

	return res
}

func (t *Trainer) Validate() result {

	input := t.valstream.Stream()

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
				res.psum += p
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
		res.psum += r.psum
		res.iters += r.iters
	}
	res.wsum = t.valstream.cache.WeightsSum()

	return res
}

func (t *Trainer) PrintSummary() {
	t.model.DecisionSummary()
	log.Println(t.streamer.cache)
	log.Println(t.valstream.cache)
}
