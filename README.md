# gosqa

[![CI](https://github.com/floatdrop/gosqa/actions/workflows/ci.yml/badge.svg)](https://github.com/floatdrop/gosqa/actions/workflows/ci.yml)

Speech quality analysis in Go using the pretrained
[deepvk/NISQA-s](https://github.com/deepvk/NISQA-s) model. Inference runs inside
your Go process with ONNX Runtime. No Python process, network request, or model
download is needed at runtime. The exported model (2.6 MB) is embedded in the
binary.

Returns five raw quality ratings: MOS, noisiness, discontinuity, coloration,
and loudness. Higher means better quality for every dimension. Predictions
are not clamped; they can fall outside the nominal training scale.

## Requirements

- Go 1.27 or newer with `CGO_ENABLED=1` and a C compiler.
- The ONNX Runtime **1.31.x CPU shared library** for the target OS/architecture.
  Tested here with 1.31.1 on macOS arm64. The pinned Go binding uses the 1.31
  C API; older runtimes are unsupported.
- **Mono float32 PCM, 48 kHz, normalized to [-1, 1].** Decode compressed audio,
  downmix, and resample before scoring. This package does not decode Opus.

Obtain the runtime from the
[official ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).
Pass the full path to the actual `.so`, `.dylib`, or `.dll` through
`Options.RuntimeLibrary` or `ONNXRUNTIME_SHARED_LIBRARY_PATH`. Ship that library
beside the relay binary or in its container image; this is a CGO application,
so a scratch image containing only the Go binary is insufficient.

## Streaming use

```go
model, err := gosqa.Open(gosqa.Options{
    RuntimeLibrary: "/opt/onnxruntime/lib/libonnxruntime.so.1.31.1",
    IntraOpThreads: 1,
})
if err != nil {
    return err
}
defer model.Close()

stream, err := model.NewStream(0) // default: 2-second frames
if err != nil {
    return err
}

// Call for every decoded mono PCM packet from this audio source.
scores, err := stream.Write(pcm)
if err != nil {
    return err
}
for _, score := range scores {
    fmt.Printf("MOS %.3f, noise %.3f\n", score.MOS, score.Noisiness)
}

// At end of source, score any partial frame, padded with zeros.
if score, ok, err := stream.Flush(); err != nil {
    return err
} else if ok {
    fmt.Printf("final MOS %.3f\n", score.MOS)
}
```

Import `github.com/floatdrop/gosqa`. While developing locally, add a `replace`
directive pointing to this checkout in the consuming module. No release of
this new module has been published yet.

Use `model.NewAnalyzer()` and `analyzer.Analyze(pcm)` when you already have
explicit chunks. It accepts 6,720–480,000 samples (140 ms–10 s). A stream's
frame size is configurable within the same limits via `NewStream(frameSamples)`.

One model can serve multiple concurrent streams. Each stream/analyzer has its
own LSTM state and must be used from a single goroutine at a time. No per-stream
native session is allocated, and streams need no `Close`. `model.Close()` waits
for in-flight calls and releases its session; subsequent inference fails with
`ErrClosed`. The small process-wide ONNX environment remains initialized until
exit. gosqa owns its initialization; do not separately initialize or destroy
the environment through the underlying ONNX binding. ONNX Runtime telemetry is
disabled during initialization.

`Reset` clears recurrent state and, for streams, pending PCM. Call it when
switching sources or reconnecting. `Flush` retains recurrent state. Input can
be reused after `Write` returns. Invalid samples are rejected before a write
consumes any data; if native inference fails after completed frames, `Write`
returns their scores with the error. Reset before continuing after such an
error.

## Beside the MoQ relay

The relay's audio catalog uses mono 48 kHz Opus. The intended integration is:

```text
MoQ audio objects → Opus decoder → bounded PCM queue → gosqa stream → metrics
```

Create one stream and Opus decoder per publisher audio track. Share the model
across worker goroutines. `Write` performs inference synchronously; run it in
an analysis worker so the relay's forwarding path can keep sending packets.
Choose the number of workers from measured CPU use; the runtime defaults to
one inference thread per call.

Preserve the audio timeline: missing packets should go through decoder packet
loss concealment or silence insertion before analysis. Scoring only surviving
packets hides discontinuities. On queue overflow or track discontinuity, reset
the analysis state and report the gap; do not silently concatenate audio from
both sides of a gap. The metric describes the decoded signal at the observation
point. Publisher-side analysis does not include downstream receiver losses.

The library and CLI live in this repository. Relay hooks, Opus decoding, and
Prometheus wiring are consumer integration work; the relay checkout is not
modified by this library.

## File scoring

The CLI reads raw little-endian float32 mono PCM at 48 kHz and writes one JSON
object per frame, including a zero-padded final partial frame:

```sh
go build -o gosqa ./cmd/gosqa
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so.1.31.1
ffmpeg -v error -i speech.wav -ac 1 -ar 48000 -f f32le - | ./gosqa
# Or use predecoded PCM:
./gosqa -input speech.f32
```

## Model parity and reproducibility

The model is pinned to upstream revision
[`d1d923cf6e80b1f2671223ab840fe46b96ffe919`](https://github.com/deepvk/NISQA-s/tree/d1d923cf6e80b1f2671223ab840fe46b96ffe919).
The exporter checks the checkpoint's SHA256 before loading it. The ONNX model
retains provenance in its metadata; see [NOTICE](NOTICE) for attribution.

The Go frontend follows upstream's actual behavior:

- Centered 960-point STFT, 480-sample hop, periodic Hann, reflection padding.
- Magnitude mel spectrum: 48 HTK bands, no normalization, 0–24 kHz.
- `20 log10`, amplitude floor `1e-4`, 80 dB clipping relative to the chunk peak.
- 15-bin segments, stride 3; six convolution layers, bidirectional LSTM, and
  average pooling with five linear score heads.
- Both forward and reverse recurrent states carried between chunks.

**Upstream's `ms_fmax: 20000` setting is ignored by its inference function.**
Matching it requires a 24 kHz upper bound. Replacing that behavior with 20 kHz,
using power spectra, changing chunk boundaries, resetting state every chunk,
or using a unidirectional LSTM changes predictions. The default two-second
chunks introduce two seconds of buffering before the first score.

The test vectors come from the unmodified upstream PyTorch model and
torchaudio preprocessing. They exercise consecutive two-second and shorter
chunks, plus silence. Tests compare every mel value, scores, and both recurrent
states. Tolerances allow float32 FFT/backend differences: 0.003 dB for mel
features and 0.0002 absolute for scores/state. This is numerical parity
validation, not a new calibration against human speech-quality judgments.

The original upstream `gt.wav` and `noisy.wav` samples were also checked through
the Go CLI against PyTorch: all ten two-second frames agreed within `0.000003`
per score. On an Apple M3 Pro with one ONNX thread, a two-second chunk took
approximately **18 ms** including **3.2 ms** of Go preprocessing (Go 1.27.3,
ONNX Runtime 1.31.1). This measures one stream; size a relay worker pool using
benchmarks on its target hardware and concurrent workload.

```sh
go test ./...                  # frontend checks; inference checks skip without runtime
ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/runtime go test -race ./...
ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/runtime go test -run '^$' -bench . -benchmem
go vet ./...
```

## Lint and CI

Use golangci-lint **v2.15.0**, matching CI, with the repository's `.golangci.yml`:

```sh
golangci-lint run ./...
golangci-lint fmt ./...         # apply gofmt and goimports locally
```

GitHub Actions runs lint, formatting checks, race tests, strict CGO pointer
checks, dependency verification, and the CLI build on pushes and pull requests.
Native inference tests run on Linux amd64 and macOS arm64 with both Go 1.27
and stable Go. CI downloads ONNX Runtime 1.31.1 from its official release and
verifies its SHA256 before testing. No Python setup is needed.

## Re-exporting the model

Python is needed only to reproduce the checked-in model and reference vectors:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r tools/requirements.txt
.venv/bin/python tools/export.py
```

This downloads checksum-pinned upstream source/checkpoint files into `.cache`,
exports ONNX-compatible overlapping adaptive pooling, and validates the export
against the original packed-sequence model with independent recurrent state
outputs. The generated window/filterbank tables preserve torchaudio's float32
constants; Go performs FFT, mel projection, dB conversion, and segmentation.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
