package ftrl

import (
	"testing"

	ml "github.com/go-code/ml/utils"
)

func BenchmarkSampleProcessing(b *testing.B) {
	fileDir := "/Users/sergey/Downloads/"
	trainFile := fileDir + "train_dataset.svm"

	df := ml.MakeAndLoadDataset(trainFile, 1)
	sample := df.Row(0)
	label := df.Label(0)

	params := MakeParams(0.1, 0.5, 0.0, 1.0, 0.5, 0.0, 1e-4, 2, 'b')
	logreg := MakeFTRL(params)
	logreg.initWeights(df.NCols())

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		processSample(logreg, sample, label, 1.0)
	}
}
