package ftrl

import (
	"log"
	"sync"
)

const (
	TrainOutputTemplate = "#%d. tr.loss=%f"
	ValOutputTemplate   = "#%02d. tr.loss=%f val.loss=%f avg(pCTR)=%f"
)

type Trainer struct {
	model     *FTRL
	streamer  *Streamer
	valstream *Streamer
	iters     uint32
}

func MakeTrainer(model *FTRL, trainStream *Streamer, valStream *Streamer, numEpoch uint32) *Trainer {
	return &Trainer{
		model:     model,
		streamer:  trainStream,
		valstream: valStream,
		iters:     numEpoch,
	}
}

func (t *Trainer) Train() {
	for i := 0; i < int(t.iters); i++ {
		input := t.streamer.Stream()

		loss := make(chan float64)
		go func() {
			t.model.Fit(input, loss)
		}()

		var wg sync.WaitGroup
		wg.Add(1)
		sumloss := 0.0
		go func(w *sync.WaitGroup) {
			for ll := range loss {
				sumloss += ll
			}
			w.Done()
		}(&wg)

		go func() {
			wg.Wait()
			log.Printf(TrainOutputTemplate, i+1, sumloss)
		}()

	}
}
