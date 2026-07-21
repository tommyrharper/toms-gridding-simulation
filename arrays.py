import numpy as np
from pathlib import Path

C = 299792458.0
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# site (lat, lon) in degrees for LOC configs -- cannot be derived from local x,y,z
_LOC_SITE = {
    "ALMA":    (-23.0229, -67.7549), "ACA": (-23.0229, -67.7549),
    "ALMASD":  (-23.0229, -67.7549), "SMA": (19.8243, -155.4783),
    "MEERKAT": (-30.7130,  21.4430),
}

def list_arrays():
    """All available configuration names (file stems in configs/)."""
    return sorted(p.name[:-4] for p in CONFIG_DIR.glob("*.cfg"))

def config_path(name):
    """Resolve 'vla.a' | 'vla.a.cfg' " a full path -> existing .cfg path."""
    p = Path(name)
    if p.suffix == ".cfg" and p.exists():
        return p
    cand = CONFIG_DIR / (name if name.endswith(".cfg") else name + ".cfg")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"config '{name}' not found in {CONFIG_DIR}")

