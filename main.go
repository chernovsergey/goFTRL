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
	pProf    = "bench.pprof"
	pTrace   = "bench.trace"
	pProfMem = "bench.mem.pprof"

	filesFolder = "./files"

	// Small dataset files
	smallDF = filesFolder + "/dataset_small"
	train   = smallDF + "/train_dataset.svm"
	trainW  = smallDF + "/weights_train.csv"
	trainF  = smallDF + "/feature_names.csv"

	valid  = smallDF + "/valid_dataset.svm"
	validW = smallDF + "/weights_valid.csv"
	validF = smallDF + "/feature_names.csv"

	modelName = "trainded.moldel"
)

func main() {
	// TODO add model serialization/deserialization
	// TODO add warmstart
	log.Println()
	train := flag.String("-t", train, "Path to TRAIN data")
	trainW := flag.String("-tw", trainW, "Path to TRAIN sample weights")
	trainF := flag.String("-tf", trainF, "Path to TRAIN feature names")
	trainR := flag.Int("-tnrows", -1, "Use at most N rows of TRAIN dataset. Below zerof")
	trainA := flag.Int("-talloc", 9500000, "Preallocate memory for N train observations")

	valid := flag.String("-v", valid, "path to VALID data")
	validW := flag.String("-vw", validW, "path to VALID weights file")
	validF := flag.String("-vf", validF, "path to VALID feature names")
	validR := flag.Int("-vnrows", -1, "Use at most N rows of VALID dataset")
	validA := flag.Int("-valloc", 5000000, "Preallocate memory for N validation observations")

	alpha := flag.Float64("-a", 0.15, "alpha")
	beta := flag.Float64("-b", 1.0, "beta")
	l1 := flag.Float64("-l1", 0.5, "L1")
	l2 := flag.Float64("-l2", 1.0, "L2")
	clip := flag.Float64("-clip", 1000.0, "gradient clip value")
	tol := flag.Float64("-tol", 1e-4, "tolerance")

	nEpoch := flag.Uint64("-e", 10, "number of epochs to train")
	bench := flag.Bool("-pprof", false, "enable profiling")
	modelfile := flag.String("-model", modelName, "Path for saving model")
	warmstart := flag.String("-warm", modelName, "Path to saved model")

	flag.Parse()

	var cpuprof, memprof, traceprof *os.File
	var err error
	if *bench {
		log.Println("pprof enabled!")

		cpuprof, err = os.Create(pProf)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		pprof.StartCPUProfile(cpuprof)

		traceprof, err = os.Create(pTrace)
		if err != nil {
			log.Fatal("could not create TRACE profile: ", err)
		}
		trace.Start(traceprof)

		memprof, err = os.Create(pProfMem)
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

	logreg := ftrl.MakeFTRL(params, *warmstart)

	dtrain := ftrl.NewDataReader(*train, *trainW, *trainF, uint32(*trainA), uint32(*trainR))
	dvalid := ftrl.NewDataReader(*valid, *validW, *validF, uint32(*validA), uint32(*validR))

	trainer := ftrl.NewTrainer(logreg, dtrain, dvalid, uint32(*nEpoch))
	trainer.Run()
	trainer.PrintSummary()

	if *modelfile != "" {
		logreg.Save(*modelfile, dtrain.GetData().NCols())
	}
}
