package ftrl

import (
	"fmt"
	"math"

	"encoding/json"

	ml "github.com/go-code/goFTRL/utils"
)

type weights struct {
	ni float64 `json: "ni"`
	zi float64 `json: "zi"`
	wi float64 `json: "wi"`
	mu ml.Locker
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

	w.mu.Lock()
	w.wi = wi
	w.mu.Unlock()

	return wi
}

func (w *weights) set(zi, ni float64) {
	w.mu.Lock()
	w.zi = zi
	w.ni = ni
	w.mu.Unlock()
}

func (w *weights) String() string {
	return fmt.Sprintf("%v\t%v", w.ni, w.zi)
}

func (w *weights) toJSON() (string, error) {
	b, err := json.Marshal(w)
	return string(b), err
}
