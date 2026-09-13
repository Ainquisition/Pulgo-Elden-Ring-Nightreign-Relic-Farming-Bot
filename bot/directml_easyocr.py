"""DirectML inference adapter for EasyOCR.

This keeps EasyOCR's existing preprocessing, box post-processing, CTC decoder,
and public ``Reader.readtext`` API, but replaces the two neural-network forward
passes with ONNX Runtime's DirectML Execution Provider.

Intended target: Windows 10/11 + a DirectX 12 GPU (tested path to be validated
on the target machine).  CPU EasyOCR remains available as a fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_DML_MODEL_FILES = ("craft_mlt_25k.onnx", "english_g2.onnx")


def _resolve_directml_model_dir() -> Path:
    """Resolve bundled or source-tree DirectML model directory."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidate = base / "directml_models"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent.parent / "directml_models"


def directml_status(require_models: bool = True) -> tuple[bool, str]:
    """Return whether ONNX Runtime DirectML is usable by this adapter."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        return False, f"onnxruntime-directml is not installed: {exc}"

    try:
        providers = ort.get_available_providers()
    except Exception as exc:
        return False, f"ONNX Runtime provider query failed: {exc}"

    if "DmlExecutionProvider" not in providers:
        return False, "DmlExecutionProvider is not available in ONNX Runtime"

    if require_models:
        model_dir = _resolve_directml_model_dir()
        missing = [name for name in _DML_MODEL_FILES if not (model_dir / name).is_file()]
        if missing:
            return False, (
                "DirectML ONNX model(s) missing: " + ", ".join(missing) +
                ". Run: python tools/setup_directml.py"
            )

    return True, ""


def _session(model_path: Path):
    """Create an ORT DirectML session using the configuration required by DML."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    # DirectML EP explicitly requires sequential execution and memory patterns off.
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        device_id = int(os.environ.get("RELICBOT_DML_DEVICE", "0"))
    except ValueError:
        device_id = 0

    providers = [
        ("DmlExecutionProvider", {"device_id": str(device_id)}),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


class _OrtDetector:
    """Torch-call-compatible facade for EasyOCR's CRAFT detector."""

    def __init__(self, model_path: Path):
        self._session = _session(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def eval(self):
        return self

    def __call__(self, x):
        # EasyOCR already performs CRAFT resize/normalization and produces an
        # NCHW float tensor.  Keep that preprocessing exactly as-is.
        import torch

        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = self._session.run(None, {self._input_name: arr})
        y = np.asarray(outputs[0], dtype=np.float32)
        # easyocr.detection.test_net expects: y, feature = net(x)
        # The feature tensor is unused by the rest of EasyOCR's CRAFT path.
        return torch.from_numpy(y), None


class _OrtRecognizer:
    """Torch-call-compatible facade for EasyOCR's English gen2 CRNN."""

    def __init__(self, model_path: Path):
        self._session = _session(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def eval(self):
        return self

    def __call__(self, image, text_for_pred=None):
        # EasyOCR's AlignCollate/NormalizePAD already emits [B,1,64,W] f32
        # normalized to [-1, 1].  The ONNX export only needs that image tensor;
        # text_for_pred is an EasyOCR interface artifact and is intentionally ignored.
        import torch

        arr = image.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = self._session.run(None, {self._input_name: arr})
        logits = np.asarray(outputs[0], dtype=np.float32)
        return torch.from_numpy(logits)


def create_directml_reader(
    lang_list: Iterable[str] = ("en",),
    *,
    model_storage_directory: str | None = None,
    download_enabled: bool = True,
    verbose: bool = False,
):
    """Create an EasyOCR Reader whose model forward passes run on DirectML.

    The base reader is initialized on CPU so all existing EasyOCR preprocessing,
    post-processing and decoder state are populated exactly as upstream expects.
    Its torch detector/recognizer objects are then replaced by DirectML ONNX
    facades.  ``reader.device`` deliberately remains ``cpu``: EasyOCR moves the
    preprocessed tensors to CPU before invoking our facades, which then hand the
    NumPy buffers to ONNX Runtime/DirectML.
    """
    ok, reason = directml_status(require_models=True)
    if not ok:
        raise RuntimeError(reason)

    import easyocr

    kwargs = {
        "gpu": False,
        "verbose": verbose,
        # Quantization is wasted work because the torch models are replaced
        # immediately after Reader initialization.
        "quantize": False,
    }
    if model_storage_directory:
        kwargs["model_storage_directory"] = model_storage_directory
        kwargs["download_enabled"] = download_enabled

    reader = easyocr.Reader(list(lang_list), **kwargs)

    model_dir = _resolve_directml_model_dir()
    reader.detector = _OrtDetector(model_dir / "craft_mlt_25k.onnx")
    reader.recognizer = _OrtRecognizer(model_dir / "english_g2.onnx")
    reader._relicbot_gpu_backend = "DirectML"
    return reader
