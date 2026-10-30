package gosqa

import (
	"errors"
	"math"
	"os"
	"sync"
	"testing"
)

func testModel(t testing.TB) *Model {
	t.Helper()
	if os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH") == "" {
		t.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH to run native inference checks")
	}
	m, err := Open(Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	return m
}

func scoreValues(s Scores) []float32 {
	return []float32{s.MOS, s.Noisiness, s.Discontinuity, s.Coloration, s.Loudness}
}

func TestInferenceAgainstPyTorch(t *testing.T) {
	m := testModel(t)
	a, err := m.NewAnalyzer()
	if err != nil {
		t.Fatal(err)
	}
	vectors := references(t)
	for _, v := range vectors {
		scores, err := a.Analyze(testSignal(v.Samples, v.Index))
		if err != nil {
			t.Fatal(err)
		}
		closeFloats(t, "scores", scoreValues(scores), v.Scores, 0.0002)
		closeFloats(t, "h", a.h[:], v.H, 0.0002)
		closeFloats(t, "c", a.c[:], v.C, 0.0002)
	}
	a.Reset()
	s, err := a.Analyze(testSignal(vectors[0].Samples, vectors[0].Index))
	if err != nil {
		t.Fatal(err)
	}
	closeFloats(t, "reset", scoreValues(s), vectors[0].Scores, 0.0002)
}

func TestStreamPacketBoundariesAndFlush(t *testing.T) {
	m := testModel(t)
	s, err := m.NewStream(0)
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.NewAnalyzer()
	if err != nil {
		t.Fatal(err)
	}
	pcm := testSignal(DefaultFrameSamples*2+1234, 0)
	var got []Scores
	// Irregular packet lengths, spanning partial and complete frames.
	for off := 0; off < len(pcm); {
		n := min(1+(off*13)%70001, len(pcm)-off)
		out, err := s.Write(pcm[off : off+n])
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, out...)
		off += n
	}
	if len(got) != 2 || s.Buffered() != 1234 {
		t.Fatalf("got %d results, %d buffered", len(got), s.Buffered())
	}
	last, ok, err := s.Flush()
	if err != nil || !ok {
		t.Fatalf("flush: %v, %v", ok, err)
	}
	got = append(got, last)
	for i, result := range got {
		chunk := make([]float32, DefaultFrameSamples)
		copy(chunk, pcm[i*DefaultFrameSamples:min((i+1)*DefaultFrameSamples, len(pcm))])
		want, err := a.Analyze(chunk)
		if err != nil {
			t.Fatal(err)
		}
		closeFloats(t, "packet boundary", scoreValues(result), scoreValues(want), 1e-6)
	}
	if _, ok, err := s.Flush(); err != nil || ok {
		t.Fatalf("second flush: %v %v", ok, err)
	}
	s.Reset()
	if s.Buffered() != 0 {
		t.Fatal("reset retained PCM")
	}
	out, err := s.Write(pcm[:DefaultFrameSamples])
	if err != nil {
		t.Fatal(err)
	}
	a.Reset()
	want, err := a.Analyze(pcm[:DefaultFrameSamples])
	if err != nil {
		t.Fatal(err)
	}
	closeFloats(t, "stream reset", scoreValues(out[0]), scoreValues(want), 1e-6)
}

func TestInvalidPCMDoesNotChangeState(t *testing.T) {
	m := testModel(t)
	s, _ := m.NewStream(0)
	if _, err := s.Write(make([]float32, 960)); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 1.01, -1.01} {
		pcm := make([]float32, DefaultFrameSamples)
		pcm[len(pcm)-1] = invalid
		if _, err := s.Write(pcm); err == nil {
			t.Fatal("accepted invalid PCM")
		}
		if s.Buffered() != 960 {
			t.Fatal("invalid Write consumed PCM")
		}
	}
	for _, count := range []int{0, MinFrameSamples - 1, MaxFrameSamples + 1} {
		if _, err := s.analyzer.Analyze(make([]float32, count)); err == nil {
			t.Fatal("accepted invalid chunk size")
		}
	}
	for _, count := range []int{-1, MinFrameSamples - 1, MaxFrameSamples + 1} {
		if _, err := m.NewStream(count); err == nil {
			t.Fatal("accepted invalid frame size")
		}
	}
}

func TestConcurrentStreamsAndClose(t *testing.T) {
	m := testModel(t)
	v := references(t)[1]
	pcm := testSignal(v.Samples, v.Index)
	wantAnalyzer, _ := m.NewAnalyzer()
	want, err := wantAnalyzer.Analyze(pcm)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a, err := m.NewAnalyzer()
			if err != nil {
				t.Error(err)
				return
			}
			got, err := a.Analyze(pcm)
			if err != nil {
				t.Error(err)
				return
			}
			closeFloats(t, "concurrent", scoreValues(got), scoreValues(want), 1e-6)
		}()
	}
	wg.Wait()
	s, _ := m.NewStream(0)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewAnalyzer(); !errors.Is(err, ErrClosed) {
		t.Fatalf("NewAnalyzer after Close: %v", err)
	}
	if _, err := wantAnalyzer.Analyze(pcm); !errors.Is(err, ErrClosed) {
		t.Fatalf("Analyze after Close: %v", err)
	}
	if _, err := s.Write(pcm); !errors.Is(err, ErrClosed) {
		t.Fatalf("Write after Close: %v", err)
	}
}

func TestZeroValues(t *testing.T) {
	var m Model
	if _, err := m.NewAnalyzer(); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	var a Analyzer
	if _, err := a.Analyze(nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	var s Stream
	if _, err := s.Write(nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, _, err := s.Flush(); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	s.Reset()
}

func BenchmarkAnalyze(b *testing.B) {
	m := testModel(b)
	a, _ := m.NewAnalyzer()
	pcm := testSignal(DefaultFrameSamples, 0)
	b.ReportAllocs()
	b.SetBytes(int64(len(pcm) * 4))
	for b.Loop() {
		if _, err := a.Analyze(pcm); err != nil {
			b.Fatal(err)
		}
	}
}

func TestInferenceChunkLimits(t *testing.T) {
	m := testModel(t)
	a, err := m.NewAnalyzer()
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{MinFrameSamples, MinFrameSamples + 1, MinFrameSamples + 479, MaxFrameSamples} {
		if _, err := a.Analyze(testSignal(count, 0)); err != nil {
			t.Fatalf("%d samples: %v", count, err)
		}
	}
}

func TestCloseDuringInference(t *testing.T) {
	m := testModel(t)
	pcm := testSignal(MaxFrameSamples, 0)
	start := make(chan struct{})
	done := make(chan error, 4)
	for i := 0; i < cap(done); i++ {
		a, err := m.NewAnalyzer()
		if err != nil {
			t.Fatal(err)
		}
		go func() {
			<-start
			_, err := a.Analyze(pcm)
			done <- err
		}()
	}
	close(start)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < cap(done); i++ {
		if err := <-done; err != nil && !errors.Is(err, ErrClosed) {
			t.Fatal(err)
		}
	}
}
