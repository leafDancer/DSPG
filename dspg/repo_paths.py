"""Filesystem paths relative to the repository root (parent of the ``dspg`` package)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
