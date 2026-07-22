"""Ensure src/ is on sys.path when running scripts under macOS + Python 3.13.

Python 3.13 skips .pth files with the macOS UF_HIDDEN flag, which breaks
editable installs under Documents/. Scripts import this first so demos work
even when the editable .pth is ignored.
"""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
