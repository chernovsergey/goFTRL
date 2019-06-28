package ftrl

import (
	"log"
	"runtime"
	"sync"
	"time"

	util "github.com/go-code/goFTRL/utils"
)

const (
	TemplateTrain    = "#%02d. tr.loss=%f time{train:%v, val:%v}"
	TemplateTrainVal = "#%02d. tr.loss=%f val.loss=%f avg(pCTR)=%f time{train:%v, val:%v}"
)

type Trainer struct {
	model     *FTRL
	train     *DataReader
	valid     *DataReader
	iters     uint32
	optimized bool
}

type result struct {
	loss  float64
	wsum  float64
	psum  float64
	iters uint64
	etime time.Duration
}

func NewTrainer(model *FTRL, trainStream *DataReader, valStream *DataReader,
	numEpoch uint32) *Trainer {
	return &Trainer{
		model: model,
		train: trainStream,
		valid: valStream,
		iters: numEpoch,
	}
}

func (t *Trainer) optimizeStorage() {
	// switch t.model.weights.(type) {
	// case *WeightMap:
	// 	ts := time.Now()
	// 	t.model.weights = MakeWeightArray(t.model.weights.(*WeightMap))
	// 	t.optimized = true
	// 	log.Println("weights store = array", time.Since(ts))
	// }
}

func (t *Trainer) Run() {

	t0 := time.Now()
	t.train.Read()
	t.valid.Read()
	log.Println("Read dataset in ", time.Since(t0))

	t0 = time.Now()
	for i := 0; i < int(t.iters); i++ {

		// if i == 1 {
		// 	t.optimizeStorage()
		// }

		tres := t.Train()

		if t.valid != nil {
			vres := t.Validate()
			log.Printf(TemplateTrainVal, i+1, tres.loss, vres.loss,
				vres.psum/vres.wsum, tres.etime, vres.etime)
			continue
		}
		log.Printf(TemplateTrain, i+1, tres.loss, tres.etime)
	}
	log.Println("Total fit time:", time.Since(t0))
}

func (t *Trainer) Train() result {

	t0 := time.Now()

	dtrain := t.train.GetData()
	njobs := runtime.NumCPU()
	chunksize := dtrain.NRows() / uint64(njobs)

	models := make([]FTRL, njobs)
	stats := make([]result, njobs)
	var wg sync.WaitGroup
	for i := 0; i < njobs; i++ {
		models[i] = t.model.Copy()

		start := chunksize * uint64(i)
		end := start + chunksize
		if end > dtrain.NRows() {
			end = dtrain.NRows()
		}

		wg.Add(1)

		go func(s uint64, e uint64, m *FTRL, stat *result, wg *sync.WaitGroup) {
			for j := s; j < e; j++ {
				o := dtrain.Row(j)
				ll := m.Fit(o)
				stat.loss += ll
				stat.wsum += o.W
			}
			wg.Done()
		}(start, end, &models[i], &stats[i], &wg)
	}
	wg.Wait()

	// Merge weights of models
	wmap := make(map[uint32]*weights)
	ncols := uint32(dtrain.NCols())
	var c uint32
	for c = 0; c < ncols; c++ {
		var newW weights
		cnt := 1
		for _, model := range models {
			w, ok := model.weights[c]
			if !ok {
				continue
			}
			cnt++
			newW.ni += w.ni
			newW.zi += w.zi
		}
		newW.ni /= float64(cnt)
		newW.zi /= float64(cnt)
		newW.get(t.model.params)
		wmap[c] = &newW
	}
	t.model.weights = wmap

	var res result
	for _, s := range stats {
		res.loss += s.loss
		res.wsum += s.wsum
	}
	res.loss /= res.wsum
	res.etime = time.Since(t0)

	return res
}

func (t *Trainer) Validate() result {

	t0 := time.Now()

	dvalid := t.valid.GetData()
	njobs := runtime.NumCPU()
	chunksize := dvalid.NRows() / uint64(njobs)

	stats := make([]result, njobs)
	var wg sync.WaitGroup
	for i := 0; i < njobs; i++ {
		start := chunksize * uint64(i)
		end := start + chunksize
		if end > dvalid.NRows() {
			end = dvalid.NRows()
		}

		wg.Add(1)

		go func(s uint64, e uint64, stat *result, wg *sync.WaitGroup) {
			for j := s; j < e; j++ {
				o := dvalid.Row(j)
				p := t.model.Predict(o.X)
				stat.loss += util.Logloss(p, o.Y, o.W)
				stat.wsum += o.W
				stat.psum += p * o.W
			}
			wg.Done()
		}(start, end, &stats[i], &wg)
	}
	wg.Wait()

	var res result
	for _, s := range stats {
		res.loss += s.loss
		res.wsum += s.wsum
		res.psum += s.psum
	}
	res.loss /= res.wsum
	res.etime = time.Since(t0)

	return res
}

func (t *Trainer) PrintSummary() {
	t.model.DecisionSummary()
	log.Println(t.train.cache)
	log.Println(t.valid.cache)
}
