package bovw

type Classifier struct {
	TrainingImage []Image
	TrainingTags  []string

	TestingImage []Image
	TestingTags  []string

	ClusterCount     int
	DescriptorsCount int

	Layout []int

	nn  *NeuralNetwork
	bow *BoW
}

func (c *Classifier) Train() {
	c.init()

	c.bow = &BoW{
		Images:           append(c.TrainingImage, c.TestingImage...),
		ClusterCount:     c.ClusterCount,
		DescriptorsCount: c.DescriptorsCount,
	}
	c.Train()

	trainingHistograms := make([]Histogram, len(c.TrainingImage))
	for i, img := range c.TrainingImage {
		trainingHistograms[i] = c.bow.Histogram(img)
	}

	testingHistograms := make([]Histogram, len(c.TestingImage))
	for i, img := range c.TestingImage {
		testingHistograms[i] = c.bow.Histogram(img)
	}

	c.nn = &NeuralNetwork{
		TrainHistograms: trainingHistograms,
		TrainTags:       c.TrainingTags,
		TestHistograms:  testingHistograms,
		TestTags:        c.TestingTags,
		Layouts:         c.Layout,
	}
	c.nn.Train()
}

func (c *Classifier) Predicate(img Image) string {
	return c.nn.Predicate(c.bow.Histogram(img))
}

func (c *Classifier) init() {
	if c.ClusterCount <= 0 {
		c.ClusterCount = 30
	}
	if c.DescriptorsCount <= 0 {
		c.DescriptorsCount = 50
	}
	if c.Layout == nil {
		c.Layout = []int{40, 20}
	}
}
