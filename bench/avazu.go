package main

import (
	"github.com/go-code/ml/ftrl"
	ml "github.com/go-code/ml/utils"
)

const (
	fileDir = "/Users/sergey/Downloads/dataset_avazu/"
)

func avazuAppModel() {

	trainFile := fileDir + "avazu-app.tr"
	validFile := fileDir + "avazu-app.val"

	Dtrain := ml.MakeAndLoadDataset(trainFile, -1, false)
	Dvalid := ml.MakeAndLoadDataset(validFile, -1, false)

	params := ftrl.MakeParams(
		0.1, 1.0, 0.5, 1.1,
		1000, 0.0, 1e-4,
		10, 'b')
	logreg := ftrl.MakeFTRL(params)
	logreg.Train(Dtrain, Dvalid)
	logreg.DecisionSummary()
}

func avazuSiteModel() {

	trainFile := fileDir + "avazu-site.tr"
	validFile := fileDir + "avazu-site.val"

	Dtrain := ml.MakeAndLoadDataset(trainFile, -1, false)
	Dvalid := ml.MakeAndLoadDataset(validFile, -1, false)

	params := ftrl.MakeParams(
		0.1, 1.0, 0.5, 1.1,
		1000, 0.0, 1e-4,
		10, 'b')
	logreg := ftrl.MakeFTRL(params)
	logreg.Train(Dtrain, Dvalid)
	logreg.DecisionSummary()
}

func main() {
	avazuAppModel()
	avazuSiteModel()
}
