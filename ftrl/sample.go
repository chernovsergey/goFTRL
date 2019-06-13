package ftrl

type Feature struct {
	Key   uint32
	Value float64
}

type Sample []Feature

type Observation struct {
	X Sample
	Y uint8
	W float64
}

type DataStream chan Observation

// LinkFunction is an alias for activation function signature
type LinkFunction func(float64) float64
