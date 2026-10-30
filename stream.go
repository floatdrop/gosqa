package gosqa

import "fmt"

// Stream buffers arbitrary PCM packets into fixed inference frames. It owns
// one frame of PCM and independent recurrent state. Use it from one goroutine
// at a time; inference is synchronous and applies backpressure to the caller.
type Stream struct {
	analyzer *Analyzer
	buffer   []float32
	used     int
}

// NewStream creates a stream. A frameSamples of zero selects the upstream
// two-second default. Other lengths must be in [MinFrameSamples, MaxFrameSamples].
func (m *Model) NewStream(frameSamples int) (*Stream, error) {
	if frameSamples == 0 {
		frameSamples = DefaultFrameSamples
	}
	if frameSamples < MinFrameSamples || frameSamples > MaxFrameSamples {
		return nil, fmt.Errorf("gosqa: frame length outside [%d, %d] samples", MinFrameSamples, MaxFrameSamples)
	}
	a, err := m.NewAnalyzer()
	if err != nil {
		return nil, err
	}
	return &Stream{analyzer: a, buffer: make([]float32, frameSamples)}, nil
}

// Write accepts normalized, mono 48 kHz PCM and returns one result per complete
// frame. Input is copied; the caller may immediately reuse the input slice.
// Invalid PCM is rejected before any input is consumed. If inference fails,
// earlier completed results are returned with the error. Reset before resuming
// after an inference error, because the rest of that Write was not consumed.
func (s *Stream) Write(pcm []float32) ([]Scores, error) {
	if s.analyzer == nil {
		return nil, ErrClosed
	}
	if err := validatePCM(pcm); err != nil {
		return nil, err
	}
	// Check even an empty or partial Write after closing the model.
	s.analyzer.model.mu.RLock()
	closed := s.analyzer.model.session == nil
	s.analyzer.model.mu.RUnlock()
	if closed {
		return nil, ErrClosed
	}
	var results []Scores
	for len(pcm) > 0 {
		n := copy(s.buffer[s.used:], pcm)
		s.used += n
		pcm = pcm[n:]
		if s.used == len(s.buffer) {
			scores, err := s.analyzer.Analyze(s.buffer)
			if err != nil {
				return results, err
			}
			results = append(results, scores)
			s.used = 0
		}
	}
	return results, nil
}

// Flush zero-pads and scores the final partial frame, matching upstream's file
// inference. ok is false when there is no buffered PCM. It retains recurrent
// state; call Reset when beginning a new source. Repeated Flush is a no-op.
func (s *Stream) Flush() (scores Scores, ok bool, err error) {
	if s.analyzer == nil {
		return Scores{}, false, ErrClosed
	}
	if s.used == 0 {
		return Scores{}, false, nil
	}
	clear(s.buffer[s.used:])
	scores, err = s.analyzer.Analyze(s.buffer)
	if err != nil {
		return Scores{}, false, err
	}
	s.used = 0
	return scores, true, nil
}

// Reset discards buffered PCM and recurrent state for a new source.
func (s *Stream) Reset() {
	if s.analyzer != nil {
		s.analyzer.Reset()
	}
	clear(s.buffer)
	s.used = 0
}

// Buffered returns the number of PCM samples waiting for a complete frame.
func (s *Stream) Buffered() int { return s.used }
