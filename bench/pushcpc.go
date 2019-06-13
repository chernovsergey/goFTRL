package main

import (
	ml "github.com/go-code/goFTRL/utils"
)

const (
	fileDir = "/Users/sergey/Downloads/dataset_pushhuge/"
)

func pushhuge() {

	train := fileDir + "train_dataset_huge.svm"
	trainW := fileDir + "weights_train_huge.csv"

	valid := fileDir + "valid_dataset_huge.svm"
	validW := fileDir + "weights_valid_huge.svm"

	_ = ml.LoadDatasetSparse(train, trainW, "", -1, true, false)
	_ = ml.LoadDatasetSparse(valid, validW, "", -1, true, false)

	// Train model
	// params := ftrl.MakeParams(
	// 	0.15, 1.0, 0.5, 1.0,
	// 	1000, 0.0, 1e-4,
	// 	10, 'b')

	// logreg := ftrl.MakeFTRL(params)
	// logreg.Fit(Dtrain, Dvalid)

	// p := logreg.PredictBatch(Dvalid)
	// log.Println(ml.Mean(p))

	// logreg.DecisionSummary()
}

func main() {
	pushhuge()
}
