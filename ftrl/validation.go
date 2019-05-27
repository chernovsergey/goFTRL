package ftrl

import (
	"runtime"

	ml "github.com/go-code/ml/utils"
)

func validateBatch(start, end int, valid *ml.Dataset, a *FTRL,
	losses chan float64, predics chan float64) {
	sumLoss := 0.0
	sumPred := 0.0
	for j := start; j < end; j++ {
		idx := uint64(j)
		x := valid.Row(idx)
		p := a.Predict(x)
		y := valid.Label(idx)
		w := valid.SampleWeight(idx)
		loss := ml.Logloss(p, y, w)
		sumLoss += loss
		sumPred += p
	}
	losses <- sumLoss
	predics <- sumPred
}

// Validate performs parallel batch iteration through
// the dataset. Computes logloss and avg. predicted
// probability
func (a *FTRL) Validate(valid *ml.Dataset) (float64, float64) {
	nrows := valid.NRows()

	nworkers := runtime.NumCPU()
	chunksize := int(nrows) / nworkers

	losses := make(chan float64, nworkers)
	predics := make(chan float64, nworkers)
	for i := 0; i < nworkers; i++ {
		start := i * chunksize
		end := start + chunksize
		if uint64(end) > nrows {
			end = int(nrows)
		}
		go validateBatch(start, end, valid, a, losses, predics)
	}

	lossSum := 0.0
	pSum := 0.0
	for i := 0; i < nworkers; i++ {
		lossSum += <-losses
		pSum += <-predics
	}

	avPCTR := pSum / float64(nrows)
	avLoss := lossSum / valid.WeightsSum()
	// PE := avPCTR/valid.MeanTarget() - 1.0
	return avLoss, avPCTR
}
