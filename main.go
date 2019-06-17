package main

import (
	"flag"
	"log"
	"os"
	"runtime/pprof"
	"runtime/trace"

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

	train := flag.String("-t", "./files/train_dataset.svm", "Path to TRAIN data")
	trainW := flag.String("-tw", "./files/weights_train.csv", "Path to TRAIN weights file")
	trainF := flag.String("-tf", "", "Path to TRAIN feature names")
	trainR := flag.Int("-tnrows", -1, "Use at most N rows of TRAIN dataset")
	trainA := flag.Int("-talloc", 9500000, "Preallocate memory for N train observations")

	valid := flag.String("-v", "./files/valid_dataset.svm", "path to VALID data")
	validW := flag.String("-vw", "./files/weights_valid.csv", "path to VALID weights file")
	validF := flag.String("-vf", "", "path to VALID feature names")
	validR := flag.Int("-vnrows", -1, "Use at most N rows of VALID dataset")
	validA := flag.Int("-valloc", 5000000, "Preallocate memory for N validation observations")

	alpha := flag.Float64("-a", 0.15, "alpha")
	beta := flag.Float64("-b", 1.0, "beta")
	l1 := flag.Float64("-l1", 0.5, "L1")
	l2 := flag.Float64("-l2", 1.0, "L2")
	clip := flag.Float64("-clip", 1000.0, "gradient clip value")
	tol := flag.Float64("-tol", 1e-4, "tolerance")

	usecache := flag.Bool("-cache", true, "use dataset caching")
	nEpoch := flag.Uint64("-e", 10, "number of epochs to train")
	bench := flag.Bool("-pprof", false, "enable profiling")

	flag.Parse()

	var cpuprof, memprof, traceprof *os.File
	var err error
	if *bench {
		log.Println("pprof enabled!")

		cpuprof, err = os.Create("bench.pprof")
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		pprof.StartCPUProfile(cpuprof)

		traceprof, err = os.Create("bench.trace")
		if err != nil {
			log.Fatal("could not create TRACE profile: ", err)
		}
		trace.Start(traceprof)

		memprof, err = os.Create("bench.mem.pprof")
		if err != nil {
			log.Fatal("could not create MEM profile: ", err)
		}
	}
	defer memprof.Close()
	defer pprof.WriteHeapProfile(memprof)
	defer cpuprof.Close()
	defer pprof.StopCPUProfile()
	defer trace.Stop()

	params := ftrl.MakeParams(
		*alpha, *beta, *l1, *l2,
		*clip, 0.0, *tol,
		*nEpoch, 'b')

	logreg := ftrl.MakeFTRL(params)

	strain := ftrl.MakeStreamer(*train, *trainW, *trainF, *usecache, uint32(*trainA), uint32(*trainR))
	svalid := ftrl.MakeStreamer(*valid, *validW, *validF, *usecache, uint32(*validA), uint32(*validR))

	trainer := ftrl.MakeTrainer(logreg, strain, svalid, uint32(*nEpoch))
	trainer.Run()
	trainer.PrintSummary()
}
