"""Download the two ONNX models used by RelicBot's DirectML adapter."""

from __future__ import annotations

import pathlib
import shutil
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "directml_models"
DEST.mkdir(parents=True, exist_ok=True)

MODELS = {
    "craft_mlt_25k.onnx": (
        "https://huggingface.co/xberg-io/sceptre-craft_mlt_25k/resolve/main/"
        "craft_mlt_25k.onnx?download=true"
    ),
    "english_g2.onnx": (
        "https://huggingface.co/xberg-io/sceptre-english_g2/resolve/main/"
        "english_g2.onnx?download=true"
    ),
}


def download(url: str, path: pathlib.Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "RelicBot-DirectML-setup/1"})
    with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", "0") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r  {path.name}: {copied / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB", end="")
        if total:
            print()
    tmp.replace(path)


print(f"DirectML model directory: {DEST}")
for name, url in MODELS.items():
    path = DEST / name
    if path.is_file() and path.stat().st_size > 1024 * 1024:
        print(f"[OK] {name} already exists ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
        continue
    print(f"Downloading {name}...")
    try:
        download(url, path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to download {name}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] {name} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")

print("DirectML models are ready.")
