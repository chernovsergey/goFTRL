package utils

import "math"

const eps float64 = 1e-15

func Logloss(p float64, y uint8, w float64) float64 {
	p = math.Max(eps, math.Min(1-eps, p))
	if y == 1 {
		return -math.Log(p) * w
	}

	return -math.Log(1.0-p) * w
}

func Norm(vec []float64) float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v * v
	}

	return math.Sqrt(sum)
}

func Mean(vec []float64) float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v // * v
	}

	return sum / float64(len(vec))
}

func InfNorm(vec []float64) float64 {
	max := -math.MaxFloat64
	for _, v := range vec {
		max = math.Max(max, math.Abs(v))
	}

	return max
}

func Clip(v float64, bound float64) float64 {
	if v > bound {
		return bound
	}

	if v < -bound {
		return -bound
	}

	return v
}
