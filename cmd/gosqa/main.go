// gosqa scores raw little-endian float32 mono PCM at 48 kHz from stdin or a file.
package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/floatdrop/gosqa"
)

func run() (runErr error) {
	runtimePath := flag.String("runtime", "", "ONNX Runtime shared library (or ONNXRUNTIME_SHARED_LIBRARY_PATH)")
	inputPath := flag.String("input", "-", "raw f32le mono 48 kHz PCM file; - reads stdin")
	frameSamples := flag.Int("frame-samples", gosqa.DefaultFrameSamples, "PCM samples per inference chunk")
	flag.Parse()
	model, err := gosqa.Open(gosqa.Options{RuntimeLibrary: *runtimePath})
	if err != nil {
		return err
	}
	defer func() { runErr = errors.Join(runErr, model.Close()) }()
	stream, err := model.NewStream(*frameSamples)
	if err != nil {
		return err
	}
	input := os.Stdin
	if *inputPath != "-" {
		input, err = os.Open(*inputPath)
		if err != nil {
			return err
		}
		defer func() { runErr = errors.Join(runErr, input.Close()) }()
	}
	encoder := json.NewEncoder(os.Stdout)
	var buf [4096 * 4]byte
	var pcm [4096]float32
	for {
		n, readErr := io.ReadFull(input, buf[:])
		if readErr != nil && !errors.Is(readErr, io.EOF) && !errors.Is(readErr, io.ErrUnexpectedEOF) {
			return readErr
		}
		if n%4 != 0 {
			return fmt.Errorf("incomplete float32 sample at end of input")
		}
		for i := range n / 4 {
			pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		scores, err := stream.Write(pcm[:n/4])
		if err != nil {
			return err
		}
		for _, score := range scores {
			if err := encoder.Encode(score); err != nil {
				return err
			}
		}
		if readErr != nil {
			break
		}
	}
	if score, ok, err := stream.Flush(); err != nil {
		return err
	} else if ok {
		return encoder.Encode(score)
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
