package main

import (
	"flag"
	"log"
	"os"
	"runtime/pprof"

	"github.com/go-code/goFTRL/ftrl"
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
	trainR := flag.Int("-tnrows", 10000000, "Use at most N rows of TRAIN dataset")

	// valid := flag.String("-v", "./files/valid_dataset.svm", "path to VALID data")
	// validW := flag.String("-vw", "./files/weights_valid.csv", "path to VALID weights file")
	// validF := flag.String("-vf", "", "path to VALID feature names")
	// validR := flag.Int("-vnrows", -1, "Use at most N rows of VALID dataset")

	alpha := flag.Float64("-a", 0.15, "alpha")
	beta := flag.Float64("-b", 1.0, "beta")
	l1 := flag.Float64("-l1", 0.5, "L1")
	l2 := flag.Float64("-l2", 1.0, "L2")
	clip := flag.Float64("-clip", 1000.0, "gradient clip value")
	tol := flag.Float64("-tol", 1e-4, "tolerance")

	usecache := flag.Bool("-cache", true, "use dataset caching")
	prealloc := flag.Uint64("-preallocN", 9500000, "preallocate memory for N observations")
	nEpoch := flag.Uint64("-e", 10, "number of epochs to train")
	bench := flag.Bool("-pprof", false, "enable profiling")

	flag.Parse()

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

	params := ftrl.MakeParams(
		*alpha, *beta, *l1, *l2,
		*clip, 0.0, *tol,
		*nEpoch, 'b')

	logreg := ftrl.MakeFTRL(params)

	strain := ftrl.MakeStreamer(*train, *trainW, *trainF, *usecache, uint32(*prealloc), uint32(*trainR))

	var svalid *ftrl.Streamer
	// svalid := ftrl.MakeStreamer(*valid, *validW, *validF, *usecache, uint32(*prealloc), uint32(*validR))

	trainer := ftrl.MakeTrainer(logreg, strain, svalid, uint32(*nEpoch))
	trainer.Run()
	logreg.DecisionSummary()
}
