package bovw

import (
	. "github.com/Hu13er/vision"
	"github.com/bugra/kmeans"
)

type Point struct {
	X, Y int
}

type Histogram []float64

func (h Histogram) Inc(i int) {
	h[i]++
}

func (h Histogram) Dim() int {
	return len(h)
}

func (h Histogram) vector() []float64 {
	return h
}

type Descriptor []float64

func (f Descriptor) Add(a Descriptor) {
	for i := range f {
		f[i] += a[i]
	}
}

func (f Descriptor) Mult(c float64) {
	for i := range f {
		f[i] *= c
	}
}

func (f Descriptor) Distance(f2 Descriptor) float64 {
	d, err := kmeans.EuclideanDistance(f.vector(), f2.vector())
	if err != nil {
		panic(err)
	}
	return d
}

func (f Descriptor) vector() []float64 {
	return f
}

type Sifter interface {
	Descriptors(n int) ([]Point, []Descriptor)
}

type Tagger interface {
	Tag() string
}

type Image interface {
	Sifter
	Matrix
}
