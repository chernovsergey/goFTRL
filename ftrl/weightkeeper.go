package ftrl

import (
	"fmt"
	"math"
)

type WeightKeeper interface {
	Get(uint32) (*weights, bool)
	Set(uint32, *weights)
	Size() uint32
	Summary() (uint32, float64, float64)
}

type WeightMap struct {
	store map[uint32]*weights
}

func MakeWeightMap() *WeightMap {
	return &WeightMap{store: make(map[uint32]*weights)}
}

func (w *WeightMap) Get(k uint32) (*weights, bool) {
	ptr, ok := w.store[k]
	return ptr, ok
}

func (w *WeightMap) Set(k uint32, weight *weights) {
	w.store[k] = weight
}

func (w *WeightMap) Size() uint32 {
	return uint32(len(w.store))
}

func (w *WeightMap) Summary() (uint32, float64, float64) {
	var countNonzero uint32
	var min, max float64
	for _, w := range w.store {
		if w.wi != 0.0 {
			countNonzero++
		}
		min = math.Min(min, w.wi)
		max = math.Max(max, w.wi)
	}
	return countNonzero, min, max
}

type WeightArray struct {
	store []*weights
}

func MakeWeightArray(m *WeightMap) *WeightArray {
	var maxkey uint32
	for k, _ := range m.store {
		if k > maxkey {
			maxkey = k
		}
	}

	arr := make([]*weights, maxkey+1)
	for k, v := range m.store {
		arr[k] = v
	}
	return &WeightArray{store: arr}
}

func (w *WeightArray) Get(k uint32) (*weights, bool) {
	if k >= w.Size() {
		return nil, false
	}

	if w.store[k] != nil {
		return w.store[k], true
	}
	return nil, false
}

func (w *WeightArray) Set(k uint32, weight *weights) {
	if k >= w.Size() {
		panic(fmt.Sprintf("Tried to set at index %d, but only %d available", k, w.Size()-1))
	}
	if weight != nil {
		w.store[k] = weight
	}
}

func (w *WeightArray) Size() uint32 {
	return uint32(len(w.store))
}

func (w *WeightArray) Summary() (uint32, float64, float64) {
	var countNonzero uint32
	var min, max float64
	for _, w := range w.store {
		if w == nil {
			continue
		}
		if w.wi != 0.0 {
			countNonzero++
		}
		min = math.Min(min, w.wi)
		max = math.Max(max, w.wi)
	}
	return countNonzero, min, max
}
