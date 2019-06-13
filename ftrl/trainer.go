package ftrl

import (
	"log"
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

func (t *Trainer) Run() {
	if t.valstream != nil {
		t.TrainAndValidate()
	}
	t.Train()
}

func (t *Trainer) Train() {
	for i := 0; i < int(t.iters); i++ {
		input := make(DataStream, 10000)
		go func() {
			t.streamer.Stream(input)
		}()

		loss := make(chan float64, 10000)
		go func() {
			t.model.Fit(input, loss)
		}()

		sumloss := 0.0
		for ll := range loss {
			sumloss += ll
		}
		// sumloss /= t.streamer.cache.weightSum
		log.Printf(TrainOutputTemplate, i, sumloss)
	}
}

func (t *Trainer) Validate() {
}

func (t *Trainer) TrainAndValidate() {
}
