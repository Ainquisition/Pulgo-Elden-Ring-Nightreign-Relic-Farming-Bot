"""Smoke-test ONNX Runtime DirectML and both RelicBot OCR models."""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "directml_models"

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit(f"[FAIL] onnxruntime import: {exc}")

print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())
if "DmlExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("[FAIL] DmlExecutionProvider is not available")

so = ort.SessionOptions()
so.enable_mem_pattern = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = [("DmlExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"]

cases = [
    ("craft_mlt_25k.onnx", np.zeros((1, 3, 128, 128), dtype=np.float32)),
    ("english_g2.onnx", np.zeros((1, 1, 64, 256), dtype=np.float32)),
]

for filename, inp in cases:
    path = MODEL_DIR / filename
    if not path.is_file():
        raise SystemExit(f"[FAIL] Missing {path}; run tools/setup_directml.py")
    print(f"Loading {filename}...")
    session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    print("  session providers:", session.get_providers())
    name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    out = session.run(None, {name: inp})
    dt = (time.perf_counter() - t0) * 1000
    shapes = [tuple(np.asarray(x).shape) for x in out]
    print(f"  [OK] first inference: {dt:.1f} ms; outputs={shapes}")

print("[PASS] DirectML provider and both OCR models executed successfully.")
