"""Export the pinned NISQA-s checkpoint and independent PyTorch test vectors.

Python is a build-time dependency only. Run from any directory:
    python tools/export.py [--cache /path/to/cache]
The default checkpoint is downloaded and SHA256-verified before unpickling.
Never replace it with an untrusted pickle.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import urllib.request
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
REVISION = "d1d923cf6e80b1f2671223ab840fe46b96ffe919"
FILES = {
    "src/core/model_torch.py": "9354da0e5f009b2b43503cd3dac9b6ac3bbef057814e84d4269396a68470c2e7",
    "src/utils/process_utils.py": "65c095af8c799963321f6b6d89e082f85ab98ef8f167965049b4de98a0141c7e",
    "src/weights/nisqa_s.tar": "39683dfcded4c8c91c9a96380d6cdf8d67848bfed7fab5a21ae20af392ddb3de",
}


def fetch(cache, name, digest):
    dest = cache / Path(name).name
    if not dest.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/deepvk/NISQA-s/{REVISION}/{name}", dest
        )
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch: {dest}")
    return dest


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adaptive_pool(x, input_width, width):
    # Height divides exactly for the default model. Width does not: implement
    # PyTorch's overlapping floor/ceil bins explicitly for ONNX export.
    x = F.max_pool2d(x, (2, 1))
    columns = [
        x[:, :, :, math.floor(i * input_width / width):math.ceil((i + 1) * input_width / width)]
        .amax(dim=3, keepdim=True)
        for i in range(width)
    ]
    return torch.cat(columns, dim=3)


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, segments, h, c):
        cnn = self.model.cnn
        x = F.relu(cnn.bn1(cnn.conv1(segments)))
        x = adaptive_pool(x, 15, 7)
        x = F.relu(cnn.bn2(cnn.conv2(x)))
        x = adaptive_pool(x, 7, 5)
        x = F.relu(cnn.bn3(cnn.conv3(x)))
        x = F.relu(cnn.bn4(cnn.conv4(x)))
        x = adaptive_pool(x, 5, 3)
        x = F.relu(cnn.bn5(cnn.conv5(x)))
        x = F.relu(cnn.bn6(cnn.conv6(x)))
        x = x.reshape(1, -1, 384)
        x, (h, c) = self.model.time_dependency.lstm(x, (h, c))
        mean = x.mean(dim=1)
        scores = torch.cat([p.linear(mean) for p in self.model.pool_layers], dim=1)
        return scores, h, c


def signal(count, index):
    # Integer arithmetic gives precisely identical float32 PCM in Go/Python.
    i = np.arange(count, dtype=np.int64)
    values = ((i * 193 + index * 997) % 65536 - 32768).astype(np.float32) / 131072
    if index == 2:
        values *= 0  # silence, including the top_db floor
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "upstream")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    files = {name: fetch(args.cache, name, digest) for name, digest in FILES.items()}
    upstream = load_module(files["src/core/model_torch.py"])
    preprocess = load_module(files["src/utils/process_utils.py"])
    torch.set_num_threads(1)
    checkpoint = torch.load(files["src/weights/nisqa_s.tar"], map_location="cpu", weights_only=False)
    model = upstream.NISQA_DIM(**checkpoint["model_args"]).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    exported = ExportModel(model).eval()
    dest = ROOT / "models" / "nisqa-s.onnx"
    dest.parent.mkdir(exist_ok=True)
    frontend = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000, n_fft=960, hop_length=480, win_length=960,
        n_mels=48, power=1.0, norm=None, mel_scale="htk",
    )
    # Preserve exact float32 constants used by torchaudio, including rounding
    # in linspace/pow/hann_window. Go performs the actual DSP independently.
    (dest.parent / "frontend.bin").write_bytes(
        frontend.spectrogram.window.numpy().astype("<f4").tobytes()
        + frontend.mel_scale.fb.numpy().astype("<f4").tobytes()
    )
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            exported,
            (torch.zeros(63, 1, 48, 15), torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)),
            str(dest),
            input_names=["segments", "h", "c"],
            output_names=["scores", "hn", "cn"],
            dynamic_axes={"segments": {0: "segments"}},
            opset_version=17,
            dynamo=False,
        )
    graph = onnx.load(str(dest))
    onnx.helper.set_model_props(graph, {
        "upstream_revision": REVISION,
        "checkpoint_sha256": FILES["src/weights/nisqa_s.tar"],
        "sample_rate": "48000",
        "score_order": "mos,noi,dis,col,loud",
        "preprocessing": "torchaudio HTK magnitude mel, fmax=24000 (upstream ignores fmax)",
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(dest))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(dest), sess_options=options, providers=["CPUExecutionProvider"])
    h, c = torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)
    vectors = []
    with torch.no_grad():
        for index, count in enumerate([96000, 15360, 96000]):
            pcm = torch.from_numpy(signal(count, index))
            mel = preprocess.get_ta_melspec(pcm, 48000, 960, 480, 960, 48, 20000)
            segments, n = preprocess.segment_specs(mel)
            expected, hn, cn = model(segments[None], torch.tensor([int(n)]), h, c)
            actual = session.run(None, {"segments": segments.numpy(), "h": h.numpy(), "c": c.numpy()})
            for got, want in zip(actual, [expected.numpy(), hn.numpy(), cn.numpy()]):
                np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)
            vectors.append({
                "samples": count,
                "index": index,
                "segments": int(n),
                "mel": mel.flatten().tolist(),
                "scores": expected.flatten().tolist(),
                "h": hn.flatten().tolist(),
                "c": cn.flatten().tolist(),
            })
            h, c = hn, cn
    (ROOT / "testdata").mkdir(exist_ok=True)
    (ROOT / "testdata" / "reference.json").write_text(json.dumps({
        "upstream_revision": REVISION,
        "torch": torch.__version__,
        "vectors": vectors,
    }, separators=(",", ":")) + "\n")
    print(f"Exported {dest} ({dest.stat().st_size} bytes); PyTorch/ONNX scores and state agree.")


if __name__ == "__main__":
    main()
