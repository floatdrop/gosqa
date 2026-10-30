// Package gosqa evaluates speech quality using the pretrained deepvk NISQA-s
// model. Inference runs in-process through ONNX Runtime; Python is not required.
//
// Input is mono, normalized float32 PCM at 48 kHz. Decode Opus and resample or
// downmix before calling this package. Scores are MOS, noisiness, discontinuity,
// coloration, and loudness ratings, with higher values indicating better
// quality. These are model estimates and are deliberately not clamped.
//
// Open one Model per worker pool and use a separate Analyzer or Stream per
// audio source. A Model may be shared concurrently. An Analyzer and a Stream
// must each be used by only one goroutine at a time. Recurrent state persists
// between chunks; Reset starts a new utterance or source. Chunk boundaries
// affect predictions because the model is bidirectional within each chunk.
package gosqa
