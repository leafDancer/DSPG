"""Filesystem anchor for resolving ``results/`` and ``figures_tables/`` from any script.

``REPO_ROOT`` is the repository root (parent directory of the ``dspg`` package). Training
scripts attach ``results`` and plotting scripts attach ``figures_tables`` under this path.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
