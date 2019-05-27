package ftrl

import (
	"fmt"
	"math"

	ml "github.com/go-code/ml/utils"
)

type weights struct {
	ni, zi float64
}

func (w *weights) get(p Params) float64 {
	if math.Abs(w.zi) <= p.lambda1 {
		return 0.0
	}

	num := ml.Sgn(w.zi)
	num *= p.lambda1
	num -= w.zi

	den := math.Sqrt(w.ni)
	den += p.beta
	den /= p.alpha
	den += p.lambda2

	wi := num / den

	return wi
}

func (w *weights) String() string {
	return fmt.Sprintf("%v\t%v", w.ni, w.zi)
}
