package gosqa

import (
	_ "embed"
	"errors"
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

//go:embed models/nisqa-s.onnx
var modelData []byte

// ErrClosed is returned when inference is attempted after Model.Close.
var ErrClosed = errors.New("gosqa: model is closed")

// Options configures the native runtime. Its process-wide environment is kept
// alive until process exit; Close releases the model's session and weights.
type Options struct {
	// RuntimeLibrary is the path to libonnxruntime.so, .dylib, or .dll (1.31.x).
	// If empty, ONNXRUNTIME_SHARED_LIBRARY_PATH is used.
	RuntimeLibrary string
	// IntraOpThreads defaults to 1 to limit CPU use beside a relay.
	IntraOpThreads int
}

var runtimeState struct {
	sync.Mutex
	path  string
	ready bool
}

func initializeRuntime(path string) error {
	runtimeState.Lock()
	defer runtimeState.Unlock()
	if runtimeState.ready {
		if path != runtimeState.path {
			return errors.New("gosqa: ONNX Runtime already initialized with a different library path")
		}
		return nil
	}
	if ort.IsInitialized() {
		return errors.New("gosqa: ONNX environment already initialized outside gosqa")
	}
	if path == "" {
		return errors.New("gosqa: set Options.RuntimeLibrary or ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("gosqa: initialize ONNX Runtime: %w", err)
	}
	if err := ort.DisableTelemetry(); err != nil {
		return errors.Join(fmt.Errorf("gosqa: disable ONNX telemetry: %w", err), ort.DestroyEnvironment())
	}
	runtimeState.path, runtimeState.ready = path, true
	return nil
}

// Model owns a native session shared by independent speech streams.
// Its zero value is closed. Always call Close when finished with a model.
type Model struct {
	mu      sync.RWMutex
	session *ort.DynamicAdvancedSession
}

// Open loads the bundled, pretrained NISQA-s model into ONNX Runtime.
// Calls to Open are safe concurrently. All models use the same runtime library.
func Open(options Options) (*Model, error) {
	if options.IntraOpThreads < 0 {
		return nil, errors.New("gosqa: IntraOpThreads must be nonnegative")
	}
	if options.RuntimeLibrary == "" {
		options.RuntimeLibrary = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}
	if err := initializeRuntime(options.RuntimeLibrary); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("gosqa: session options: %w", err)
	}
	// Fresh options are destroyed exactly once; their C release cannot fail.
	defer func() { _ = opts.Destroy() }()
	threads := options.IntraOpThreads
	if threads == 0 {
		threads = 1
	}
	if err := opts.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelData,
		[]string{"segments", "h", "c"}, []string{"scores", "hn", "cn"}, opts)
	if err != nil {
		return nil, fmt.Errorf("gosqa: load model: %w", err)
	}
	return &Model{session: session}, nil
}

// Close waits for in-flight inference and releases the native model session.
// It is safe to call more than once or concurrently with inference.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session == nil {
		return nil
	}
	err := m.session.Destroy()
	m.session = nil
	return err
}

// NewAnalyzer creates independent recurrent state. No native resources are
// allocated per analyzer, and it needs no Close call.
func (m *Model) NewAnalyzer() (*Analyzer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.session == nil {
		return nil, ErrClosed
	}
	return &Analyzer{model: m, frontend: newFrontend(), h: make([]float32, 256), c: make([]float32, 256)}, nil
}
