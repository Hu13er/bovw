package bovw

func ImageSliceCacher(imgs []Image) []Image {
	outp := make([]Image, len(imgs))
	for i := range imgs {
		outp[i] = ImageCacher(imgs[i])
	}
	return outp
}

func ImageCacher(img Image) Image {
	return &SiftCacher{Image: img}
}

type SiftCacher struct {
	Image
	desc   []Descriptor
	points []Point
}

var _ Sifter = (*SiftCacher)(nil)

func (s *SiftCacher) Descriptors(n int) ([]Point, []Descriptor) {
	if s.desc != nil && s.points != nil {
		return s.points, s.desc
	}
	s.points, s.desc = s.Image.Descriptors(n)
	return s.points, s.desc
}
