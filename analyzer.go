package gosqa

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// Scores are raw model ratings in the upstream order MOS, NOI, DIS, COL, LOUD.
// Higher values indicate better quality for all five dimensions.
type Scores struct {
	MOS           float32 `json:"mos"`
	Noisiness     float32 `json:"noisiness"`
	Discontinuity float32 `json:"discontinuity"`
	Coloration    float32 `json:"coloration"`
	Loudness      float32 `json:"loudness"`
}

// Analyzer evaluates explicit PCM chunks while preserving LSTM state.
// Use one per audio source, from one goroutine at a time. Create with
// Model.NewAnalyzer. Use Stream to buffer arbitrary packet-sized inputs.
type Analyzer struct {
	model    *Model
	frontend frontend
	h, c     []float32
}

// Reset clears recurrent state, for example after reconnecting or switching
// sources. No PCM is retained by an Analyzer.
func (a *Analyzer) Reset() { clear(a.h[:]); clear(a.c[:]) }

func validatePCM(pcm []float32) error {
	for i, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < -1 || v > 1 {
			return fmt.Errorf("gosqa: PCM sample %d must be finite and within [-1, 1]", i)
		}
	}
	return nil
}

// Analyze scores one mono 48 kHz chunk, advancing recurrent state only on
// success. Length must be between MinFrameSamples and MaxFrameSamples. Match
// upstream with two-second chunks, including zero-padding the last chunk.
// Changing chunk boundaries changes predictions. Silence is scored normally.
func (a *Analyzer) Analyze(pcm []float32) (Scores, error) {
	if a.model == nil {
		return Scores{}, ErrClosed
	}
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.model.session == nil {
		return Scores{}, ErrClosed
	}
	if len(pcm) < MinFrameSamples || len(pcm) > MaxFrameSamples {
		return Scores{}, fmt.Errorf("gosqa: chunk length %d outside [%d, %d] samples", len(pcm), MinFrameSamples, MaxFrameSamples)
	}
	if err := validatePCM(pcm); err != nil {
		return Scores{}, err
	}
	segments, count := a.frontend.extract(pcm)
	scoreData, hnData, cnData := make([]float32, 5), make([]float32, 256), make([]float32, 256)
	// ONNX retains pointers to Go-owned buffers for the lifetime of its tensors.
	// Pin until all native values have been destroyed (deferred below).
	var pins runtime.Pinner
	defer pins.Unpin()
	for _, data := range [][]float32{segments, a.h, a.c, scoreData, hnData, cnData} {
		pins.Pin(&data[0])
	}
	x, err := ort.NewTensor(ort.NewShape(int64(count), 1, melBands, segmentWidth), segments)
	if err != nil {
		return Scores{}, err
	}
	// Tensor.Destroy wraps a void C release and always returns nil.
	defer func() { _ = x.Destroy() }()
	h, err := ort.NewTensor(ort.NewShape(2, 1, 128), a.h[:])
	if err != nil {
		return Scores{}, err
	}
	defer func() { _ = h.Destroy() }()
	c, err := ort.NewTensor(ort.NewShape(2, 1, 128), a.c[:])
	if err != nil {
		return Scores{}, err
	}
	defer func() { _ = c.Destroy() }()
	// Allocate known output shapes, avoiding auto-allocation and ownership
	// ambiguity if a native run fails partway through.
	output, err := ort.NewTensor(ort.NewShape(1, 5), scoreData)
	if err != nil {
		return Scores{}, err
	}
	defer func() { _ = output.Destroy() }()
	hn, err := ort.NewTensor(ort.NewShape(2, 1, 128), hnData)
	if err != nil {
		return Scores{}, err
	}
	defer func() { _ = hn.Destroy() }()
	cn, err := ort.NewTensor(ort.NewShape(2, 1, 128), cnData)
	if err != nil {
		return Scores{}, err
	}
	defer func() { _ = cn.Destroy() }()
	err = a.model.session.Run([]ort.Value{x, h, c}, []ort.Value{output, hn, cn})
	if err != nil {
		return Scores{}, fmt.Errorf("gosqa: inference: %w", err)
	}
	for _, values := range [][]float32{output.GetData(), hn.GetData(), cn.GetData()} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return Scores{}, errors.New("gosqa: nonfinite model output")
			}
		}
	}
	copy(a.h[:], hn.GetData())
	copy(a.c[:], cn.GetData())
	s := output.GetData()
	return Scores{MOS: s[0], Noisiness: s[1], Discontinuity: s[2], Coloration: s[3], Loudness: s[4]}, nil
}
