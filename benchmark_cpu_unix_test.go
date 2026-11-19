//go:build darwin || linux

package gosqa

import (
	"syscall"
	"testing"
	"time"
)

// processCPUTime counts cumulative user and system CPU time across all
// threads, including native ONNX work. The boolean reports OS support.
func processCPUTime(b *testing.B) (time.Duration, bool) {
	b.Helper()
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		b.Fatalf("read process CPU time: %v", err)
	}
	return time.Duration(usage.Utime.Nano() + usage.Stime.Nano()), true
}
