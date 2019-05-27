package main

import (
	"log"
	"os"
	"runtime/pprof"

	"github.com/go-code/ml/ftrl"
	ml "github.com/go-code/ml/utils"
)

func main() {
	//log.SetOutput(ioutil.Discard)

	// TODO START
	// Makte it as problem config
	fileDir := "/Users/sergey/Downloads/"

	trainFile := fileDir + "train_dataset.svm"
	trainWeights := fileDir + "weights_train.csv"

	validFile := fileDir + "valid_dataset.svm"
	validWeights := fileDir + "weights_valid.csv"

	//featureNames := fileDir + "feature_names.csv"
	//TODO END

	Dtrain := ml.MakeAndLoadDataset(trainFile, -1, true)
	Dtrain.LoadSampleWeights(trainWeights)

	Dvalid := ml.MakeAndLoadDataset(validFile, -1, true)
	Dvalid.LoadSampleWeights(validWeights)

	f, err := os.Create("bench.pprof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()

	params := ftrl.MakeParams(
		0.15, 1.0, 0.1, 1.0,
		1000, 0.0, 1e-4,
		10, 'b')
	logreg := ftrl.MakeFTRL(params)
	logreg.Fit(Dtrain, Dvalid)
	p := logreg.PredictBatch(Dvalid)
	log.Println(ml.Mean(p))
	logreg.DecisionSummary()
}
