package ftrl

import "fmt"

type Params struct {
	alpha, beta, lambda1, lambda2 float64
	clipgrad                      float64
	dropout                       float64
	tol                           float64
	niter                         uint64
	activation                    rune
}

func MakeParams(
	a, b, l1, l2, clipgrad, dropout, tol float64,
	maxiter uint64, activation rune) Params {
	return Params{
		alpha:      a,
		beta:       b,
		lambda1:    l1,
		lambda2:    l2,
		clipgrad:   clipgrad,
		dropout:    dropout,
		tol:        tol,
		niter:      maxiter,
		activation: activation}
}

func (p *Params) String() string {
	return fmt.Sprintf("FTRL{Alpha:%v, Beta:%v, L1:%v, L2:%v, max_iter:%v, activation:%q}",
		p.alpha, p.beta, p.lambda1, p.lambda2, p.niter, p.activation)
}
