#!/usr/bin/env python3
"""Fetch a sentence-embedding model (ONNX + tokenizer) from Hugging Face into ./models/<name>.

  python scripts/download_model.py                       # all-MiniLM-L6-v2 (English, 90 MB, default)
  python scripts/download_model.py --multilingual        # paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)

The tool runs without it (lexical zero-shot only), but zero-label accuracy is much better with it.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "all-MiniLM-L6-v2": ("Xenova/all-MiniLM-L6-v2",
                         {"model.onnx": "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
                          "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"}),
    "paraphrase-multilingual-MiniLM-L12-v2": ("Xenova/paraphrase-multilingual-MiniLM-L12-v2", {}),
}


def fetch(repo, fname, dest):
    remote = f"onnx/{fname}" if fname.endswith(".onnx") else fname
    url = f"https://huggingface.co/{repo}/resolve/main/{remote}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multilingual", action="store_true")
    a = ap.parse_args()
    name = "paraphrase-multilingual-MiniLM-L12-v2" if a.multilingual else "all-MiniLM-L6-v2"
    repo, hashes = MODELS[name]
    out = ROOT / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in ["model.onnx", "tokenizer.json"]:
        dest = out / f
        if not dest.exists():
            fetch(repo, f, dest)
        if f in hashes:
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            print(f"  {f} sha256 {h} {'OK' if h == hashes[f] else '(differs from the tested file - still usable)'}")
    print(f"model ready in {out}. Set `embedding_model: models/{name}` in a config to use another model.")


if __name__ == "__main__":
    sys.exit(main())
