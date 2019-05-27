package main

import (
	"flag"
	"log"
	"os"
	"runtime/pprof"

	"github.com/go-code/ml/ftrl"
	ml "github.com/go-code/ml/utils"
)

const (
	pProf = "bench.pprof"
)

func main() {
	train := flag.String("-t", "./files/train_dataset.svm", "path to TRAIN data")
	trainW := flag.String("-tw", "./files/weights_train.csv", "path to TRAIN weights file")
	trainF := flag.String("-tf", "", "path to TRAIN feature names")

	valid := flag.String("-v", "./files/valid_dataset.svm", "path to VALID data")
	validW := flag.String("-vw", "./files/weights_valid.csv", "path to VALID weights file")
	validF := flag.String("-vf", "", "path to VALID feature names")

	alpha := flag.Float64("-a", 0.15, "alpha")
	beta := flag.Float64("-b", 1.0, "beta")
	l1 := flag.Float64("-l1", 0.5, "L1")
	l2 := flag.Float64("-l2", 1.0, "L2")
	clip := flag.Float64("-clip", 1000.0, "gradient clip value")
	tol := flag.Float64("-tol", 1e-4, "tolerance")

	nEpoch := flag.Uint64("-e", 10, "number of epochs to train")
	bench := flag.Bool("-pprof", true, "enable profiling")

	flag.Parse()

	if *bench {
		log.Println("pprof enabled!")
	}

	f, err := os.Create("bench.pprof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()

	// Parse train
	Dtrain := ml.MakeAndLoadDataset(*train, -1, true)
	if *trainW != "" {
		Dtrain.LoadSampleWeights(*trainW)
		if *validF != "" {
			Dtrain.LoadFeatureNames(*trainF)
		}
	}

	// Parse validation
	var Dvalid *ml.Dataset
	if *valid != "" {
		Dvalid = ml.MakeAndLoadDataset(*valid, -1, true)
		if *validW != "" {
			Dvalid.LoadSampleWeights(*validW)
		}
		if *validF != "" {
			Dvalid.LoadFeatureNames(*validF)
		}
	}

	// Train model
	params := ftrl.MakeParams(
		*alpha, *beta, *l1, *l2,
		*clip, 0.0, *tol,
		*nEpoch, 'b')

	logreg := ftrl.MakeFTRL(params)
	logreg.Fit(Dtrain, Dvalid)

	p := logreg.PredictBatch(Dvalid)

	log.Println(ml.Mean(p))
	logreg.DecisionSummary()
}
