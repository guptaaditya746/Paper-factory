"""Where the repo's top folders are (one place, so a folder move changes one line)."""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1]      # every Python package
REPO = SRC.parent
DATA = REPO / "data"                           # caches (data/cache) and exports; runs go to runs/
