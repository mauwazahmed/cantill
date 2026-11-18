"""Local sentence embeddings (ONNX Runtime + HF tokenizers). No API, no torch.

Expects a folder with `model.onnx` and `tokenizer.json` (e.g. Xenova/all-MiniLM-L6-v2, fetched by
scripts/download_model.py). If the folder or the libraries are missing, `load_embedder` returns None and the
classifier falls back to lexical similarity only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir: str | Path, max_len: int = 64, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx = model_dir / "model.onnx"
        if not onnx.exists():
            onnx = model_dir / "onnx" / "model.onnx"
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.name = model_dir.name
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts, batch_size: int = 256) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        # sort by length for efficient padding
        todo.sort(key=len)
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            enc = self.tok.encode_batch(chunk)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            am = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": am}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            v = (h * am[..., None]).sum(1) / np.maximum(am.sum(1, keepdims=True), 1)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            for t, row in zip(chunk, v.astype(np.float32)):
                self._cache[t] = row
        return np.vstack([self._cache[t] for t in texts])


def load_embedder(model_dir: str | Path | None):
    if not model_dir:
        return None
    try:
        p = Path(model_dir)
        if not p.exists():
            print(f"[embed] model folder {p} not found -> lexical-only zero-shot (run scripts/download_model.py)")
            return None
        return Embedder(p)
    except Exception as e:  # noqa: BLE001
        print(f"[embed] could not load embedder ({e}) -> lexical-only zero-shot")
        return None
