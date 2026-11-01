package gosqa

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type referenceVector struct {
	Samples  int       `json:"samples"`
	Index    int       `json:"index"`
	Segments int       `json:"segments"`
	Mel      []float32 `json:"mel"`
	Scores   []float32 `json:"scores"`
	H        []float32 `json:"h"`
	C        []float32 `json:"c"`
}

func references(t testing.TB) []referenceVector {
	t.Helper()
	b, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var data struct {
		Vectors []referenceVector `json:"vectors"`
	}
	if err := json.Unmarshal(b, &data); err != nil {
		t.Fatal(err)
	}
	return data.Vectors
}

func testSignal(count, index int) []float32 {
	pcm := make([]float32, count)
	for i := range pcm {
		if index != 2 {
			pcm[i] = float32((int64(i)*193+int64(index)*997)%65536-32768) / 131072
		}
	}
	return pcm
}

func closeFloats(t testing.TB, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	var worst float64
	var worstIndex int
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatalf("%s[%d]: nonfinite", name, i)
		}
		if diff > worst {
			worst, worstIndex = diff, i
		}
	}
	if worst > tolerance {
		t.Fatalf("%s[%d]: got %.9g, want %.9g (max error %.9g > %.9g)", name, worstIndex, got[worstIndex], want[worstIndex], worst, tolerance)
	}
}

func TestFrontendAgainstTorchaudio(t *testing.T) {
	f := newFrontend()
	for _, v := range references(t) {
		segments, n := f.extract(testSignal(v.Samples, v.Index))
		if n != v.Segments {
			t.Fatalf("segments: got %d, want %d", n, v.Segments)
		}
		closeFloats(t, "mel", f.mel, v.Mel, 0.003)
		bins := v.Samples/hopSize + 1
		for s := range n {
			for m := range melBands {
				closeFloats(t, "segment", segments[(s*melBands+m)*segmentWidth:(s*melBands+m+1)*segmentWidth], v.Mel[m*bins+s*segmentHop:m*bins+s*segmentHop+segmentWidth], 0.003)
			}
		}
	}
}

func TestFrontendBoundaries(t *testing.T) {
	f := newFrontend()
	for _, count := range []int{MinFrameSamples, MinFrameSamples + 1, MinFrameSamples + 479, MaxFrameSamples} {
		segments, n := f.extract(make([]float32, count))
		if n < 1 || len(segments) != n*melBands*segmentWidth {
			t.Fatal("invalid shape")
		}
		for _, v := range segments {
			if v != -80 {
				t.Fatalf("silence: got %v, want -80", v)
			}
		}
	}
}

func BenchmarkFrontend(b *testing.B) {
	f := newFrontend()
	pcm := testSignal(DefaultFrameSamples, 0)
	b.ReportAllocs()
	b.SetBytes(int64(len(pcm) * 4))
	for b.Loop() {
		f.extract(pcm)
	}
}
