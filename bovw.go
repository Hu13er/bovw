package bovw

import (
	"github.com/bugra/kmeans"
)

type BoW struct {
	Images           []Image
	DescriptorsCount int
	ClusterCount     int

	features           [][]float64
	clustedDescriptors map[int][]Descriptor
	clusterMeans       []Descriptor
}

func (bow *BoW) Train() {
	bow.init()

	for _, img := range bow.Images {
		_, features := img.Descriptors(bow.DescriptorsCount)
		bow.features = append(bow.features, featureVector(features)...)
	}
	labels, err := kmeans.Kmeans(
		bow.features, bow.ClusterCount, kmeans.EuclideanDistance, 30)
	if err != nil {
		panic(err)
	}

	for i, labels := range labels {
		bow.clustedDescriptors[labels] = append(
			bow.clustedDescriptors[labels], bow.features[i])
	}

	bow.clusterMeans = make([]Descriptor, len(bow.clustedDescriptors))
	for k, fs := range bow.clustedDescriptors {
		mean := make(Descriptor, len(fs[0]))
		for _, f := range fs {
			mean.Add(f)
		}
		mean.Mult(1 / float64(len(fs)))
		bow.clusterMeans[k] = mean
	}
}

func (bow *BoW) init() {
	if bow.ClusterCount <= 0 {
		panic("cluster count can not be zero")
	}
	if bow.DescriptorsCount <= 0 {
		panic("features count can not be zero")
	}
	if bow.features == nil {
		bow.features = make([][]float64, 0)
	}
	if bow.clustedDescriptors == nil {
		bow.clustedDescriptors = make(map[int][]Descriptor, 0)
	}
}

func (bow *BoW) word(f Descriptor) int {
	imin, min := -1, 1e6
	for i, fi := range bow.clusterMeans {
		m := f.Distance(fi)
		if min < m {
			min = m
			imin = i
		}
	}
	return imin
}

func (bow *BoW) Histogram(img Image) Histogram {
	h := make(Histogram, bow.ClusterCount)
	_, features := img.Descriptors(bow.DescriptorsCount)
	for _, f := range features {
		nth := bow.word(f)
		h.Inc(nth)
	}
	return h
}

func featureVector(fs []Descriptor) [][]float64 {
	outp := make([][]float64, len(fs))
	for i, f := range fs {
		outp[i] = f.vector()
	}
	return outp
}
