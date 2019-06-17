package main

import (
	"github.com/go-code/goFTRL/ftrl"
)

const (
	fileDir = "/Users/sergey/Downloads/dataset_pushhuge/"
)

func pushhuge() {

	train := fileDir + "train_dataset_huge.svm"
	trainW := fileDir + "weights_train_huge.csv"

	valid := fileDir + "valid_dataset_huge.svm"
	validW := fileDir + "weights_valid_huge.csv"

	params := ftrl.MakeParams(
		0.15, 1.0, 0.5, 1.0,
		1000, 0.0, 1e-4,
		10, 'b')
	logreg := ftrl.MakeFTRL(params)

	strain := ftrl.MakeStreamer(train, trainW, "", true, uint32(114524174+1), uint32(200000000))
	svalid := ftrl.MakeStreamer(valid, validW, "", true, uint32(13302027+1), uint32(200000000))

	trainer := ftrl.MakeTrainer(logreg, strain, svalid, uint32(10))
	trainer.Run()
	trainer.PrintSummary()
}

func main() {
	pushhuge()
}
