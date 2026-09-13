# Experimental DirectML GPU path (Windows 10/11 + AMD/Intel GPUs)

This fork keeps the original CPU EasyOCR path and adds an experimental GPU path:

`EasyOCR preprocessing/postprocessing -> ONNX Runtime -> DirectML -> GPU`

Only the neural network forward passes (CRAFT detector and English gen2 CRNN recognizer)
are redirected to DirectML. EasyOCR's existing cropping, box grouping, CTC decoding and
RelicBot result format remain unchanged.

## Install from source

Use 64-bit Python 3.14, matching this repository's pinned dependencies.

```bat
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-directml.txt
python tools\setup_directml.py
python tools\check_directml.py
python main.py
```

If `check_directml.py` prints `[PASS]`, RelicBot should expose GPU Acceleration using
DirectML even when CUDA is unavailable.

## Adapter selection

- If `torch.cuda.is_available()` is true, the original EasyOCR CUDA path is preferred.
- Otherwise, if DirectML + both ONNX models are available, the DirectML adapter is used.
- Otherwise, the GPU toggle is disabled and CPU mode remains unchanged.

`RELICBOT_DML_DEVICE` can override the DirectML adapter index (default `0`).

## Build an EXE

Run `tools\setup_directml.py` before PyInstaller. The patched spec conditionally collects
ONNX Runtime and includes `directml_models/*.onnx` when they are present.

```bat
pyinstaller relic_bot.spec
```

## Validation

This DirectML path has been tested on Windows 10 with an AMD Radeon RX 9070 XT.
Control and navigation OCR intentionally remain on CPU EasyOCR; DirectML is used for
the heavier relic-analysis OCR. Run `tools\check_directml.py` first on each target PC.
It is intentionally separate from RelicBot so driver, provider, and model problems can
be diagnosed before starting the bot.
