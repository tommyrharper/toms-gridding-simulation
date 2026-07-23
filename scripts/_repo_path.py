"""Ensure src/ and the repo root are on sys.path when running scripts under
macOS + Python 3.13.

Python 3.13 skips .pth files with the macOS UF_HIDDEN flag, which breaks
editable installs under Documents/. Scripts import this first so demos work
even when the editable .pth is ignored. The repo root is included too so
top-level packages outside src/ (e.g. ml/) are importable.
"""
from pathlib import Path
import sys

_ROOT: Path = Path(__file__).resolve().parents[1]
_SRC: Path = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
