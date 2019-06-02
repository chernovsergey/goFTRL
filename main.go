package main

import (
	"flag"
	"log"
	"os"
	"runtime/pprof"

	"github.com/go-code/goFTRL/ftrl"
	ml "github.com/go-code/goFTRL/utils"
)

const (
	pProf = "bench.pprof"
)

func main() {
	// TODO add flag if to read binary dataset
	// TODO add flag read fixed number of rows
	// TODO enable profile if flag set
	// TODO add model serialization/deserialization
	// TODO add warmstart

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

	// Profiling
	var prof *os.File
	var err error
	if *bench {
		log.Println("pprof enabled!")
		prof, err = os.Create("bench.pprof")
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		if err = pprof.StartCPUProfile(prof); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
	}
	defer prof.Close()
	defer pprof.StopCPUProfile()

	// Parse train & validation
	Dtrain := ml.LoadDataset(*train, *trainW, *trainF, -1, true, false)
	Dvalid := ml.LoadDataset(*valid, *validW, *validF, -1, true, false)

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
