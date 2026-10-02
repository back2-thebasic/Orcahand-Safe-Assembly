"""Portable paths for exported metadata, relative to explicit base directories."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def relative_path(path, base=PACKAGE_ROOT):
    """Return a slash-separated relative path without changing runtime lookup."""
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
