"""Locate model assets without mixing them with source files."""
import os
from pathlib import Path


def model_root():
    configured = os.environ.get("FINECRAFT_MODEL_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1] / "weights"
