"""Saving and loading the model's brain (its learned weights)."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import GPT, GPTConfig
from .optim import AdamW


def save_checkpoint(path, model, optimizer=None, meta=None):
    path = Path(path)
    arrays = {f"param/{k}": v for k, v in model.params.items()}
    info = {"config": asdict(model.config), "meta": meta or {}}
    if optimizer is not None:
        arrays.update({f"adam_m/{k}": v for k, v in optimizer.m.items()})
        arrays.update({f"adam_v/{k}": v for k, v in optimizer.v.items()})
        info["adam_t"] = optimizer.t
    arrays["info"] = np.array(json.dumps(info))
    # Write to a temporary file first so a crash never leaves a broken checkpoint.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Returns (model, optimizer or None, meta)."""
    with np.load(path, allow_pickle=False) as f:
        info = json.loads(f["info"].item())
        model = GPT(GPTConfig(**info["config"]))
        model.params = {k: f[f"param/{k}"].astype(np.float32) for k in model.params}
        optimizer = None
        if "adam_t" in info:
            optimizer = AdamW(model.params)
            optimizer.m = {k: f[f"adam_m/{k}"].astype(np.float32) for k in model.params}
            optimizer.v = {k: f[f"adam_v/{k}"].astype(np.float32) for k in model.params}
            optimizer.t = info["adam_t"]
    return model, optimizer, info["meta"]
