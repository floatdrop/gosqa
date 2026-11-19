//go:build !darwin && !linux

package gosqa

import (
	"testing"
	"time"
)

func processCPUTime(*testing.B) (time.Duration, bool) { return 0, false }
