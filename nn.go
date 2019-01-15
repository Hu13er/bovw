package bovw

import (
	"math/rand"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

type NeuralNetwork struct {
	TrainHistograms []Histogram
	TrainTags       []string

	TestHistograms []Histogram
	TestTags       []string

	Layouts []int

	neural        *deep.Neural
	trainExamples training.Examples
	testExamples  training.Examples

	tags []string
}

func (nn *NeuralNetwork) Train() {
	trainer := training.NewBatchTrainer(training.NewAdam(0.02, 0.9, 0.999, 1e-8), 0, 200, 8)
	trainer.Train(nn.neural, nn.trainExamples, nn.testExamples, 500)
}

func (nn *NeuralNetwork) Predicate(h Histogram) string {
	output := nn.neural.Predict(h.vector())
	imax, max := -1, -1e6
	for i, n := range output {
		if max < n {
			max = n
			imax = i
		}
	}
	return nn.untag(imax)
}

func (nn *NeuralNetwork) init() {
	if len(nn.TrainHistograms) <= 0 {
		panic("train set can not be empty")
	}
	if len(nn.TrainHistograms) != len(nn.TrainTags) {
		panic("train histogram len != train tag len")
	}
	if len(nn.TestHistograms) <= 0 {
		panic("test set can not be empty")
	}
	if len(nn.TestHistograms) != len(nn.TestTags) {
		panic("test histogram len != test tag len")
	}

	if nn.neural == nil {
		inputs := nn.TrainHistograms[0].Dim()
		outputs := len(nn.TrainTags)
		layout := append(nn.Layouts, outputs)

		cnf := &deep.Config{
			Inputs:     inputs,
			Layout:     layout,
			Activation: deep.ActivationReLU,
			Mode:       deep.ModeMultiClass,
			Weight:     deep.NewNormal(0.6, 0.1),
			Bias:       true,
		}
		nn.neural = deep.NewNeural(cnf)
	}

	if nn.trainExamples == nil {
		nn.trainExamples = make(training.Examples, 0)
		for i := range nn.TrainHistograms {
			nn.trainExamples = append(nn.trainExamples,
				nn.histogramTagToExample(nn.TrainHistograms[i], nn.TrainTags[i]))
		}
	}

	if nn.testExamples == nil {
		for i := range nn.TestHistograms {
			nn.testExamples = append(nn.testExamples,
				nn.histogramTagToExample(nn.TestHistograms[i], nn.TestTags[i]))
		}
	}
}

func (nn *NeuralNetwork) histogramTagToExample(h Histogram, tag string) training.Example {
	resp := make([]float64, len(nn.tags))
	resp[nn.tag(tag)] = 1.0
	return training.Example{
		Input:    h.vector(),
		Response: resp,
	}
}

func (nn *NeuralNetwork) initTags() {
	s := map[string]struct{}{}
	for _, t := range nn.TrainTags {
		s[t] = struct{}{}
	}

	i := 0
	nn.tags = make([]string, len(s))
	for v, _ := range s {
		nn.tags[i] = v
		i++
	}
}

func (nn *NeuralNetwork) tag(t string) int {
	for i, s := range nn.tags {
		if s == t {
			return i
		}
	}
	return 0
}

func (nn *NeuralNetwork) untag(t int) string {
	return nn.tags[t]
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
