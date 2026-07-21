import numpy as np
from pathlib import Path

C = 299792458.0
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# site (lat, lon) in degrees for LOC configs -- cannot be derived from local x,y,z
_LOC_SITE = {
    "ALMA": (-23.0229, -67.7549),
    "ACA": (-23.0229, -67.7549),
    "ALMASD": (-23.0229, -67.7549),
    "SMA": (19.8243, -155.4783),
    "MEERKAT": (-30.7130, 21.4430),
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


def _read_cfg(path):
    """Parse a .cfg -> (positions [N,3] float, coordsys, observatory)."""
    coordsys, obs, rows = "XYZ", "", []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                low = s.lower()
                if "coordsys" in low:
                    coordsys = low.split("coordsys")[1].lstrip("= ").split()[0].upper()
                elif "observatory" in low:
                    obs = s.split("=")[-1].strip()
                continue
            if not s:
                continue
            p = s.split()
            rows.append([float(p[0]), float(p[1]), float(p[2])])
        return np.array(rows), coordsys, obs


def _enu_to_local_equatorial(enu, lat):
    E, N, U = enu[:, 0], enu[:, 1], enu[:, 2]
    return np.column_stack(
        [-N * np.sin(lat) + U * np.cos(lat), E, N * np.cos(lat) + U * np.sin(lat)]
    )


def antennas_local_equatorial(name):
    """Antennas in the local-equatorial fram [m], site latitude [rad],
    site longitude [deg]. WOrks for any XYZ or supported-LOC config."""
    path = config_path(name)
    xyz, coordsys, obs = _read_cfg(path)

    if coordsys == "XYZ":  # geocentric ITRF
        cx, cy, cz = xyz.mean(axis=0)
        lon = np.arctan2(cy, cx)
        lat = np.arctan2(cz, np.hypot(cx, cy))
        rel = xyz - xyz.mean(axis=0)  # subtract centre for precision
        x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
        le = np.column_stack(
            [np.cos(lon) * x + np.sin(lon) * y, -np.sin(lon) * x + np.cos(lon) * y, z]
        )
        return le, lat, np.rad2deg(lon)

    if coordsys == "LOC":  # local tangent plane E,N,U
        key = obs.upper()
        if key not in _LOC_SITE:
            raise NotImplementedError(
                f"LOC config for '{obs}' needs a site lat/lon; known: {sorted(_LOC_SITE)}"
            )
        lat_deg, lon_deg = _LOC_SITE[key]
        lat = np.deg2rad(lat_deg)
        return _enu_to_local_equatorial(xyz, lat), lat, lon_deg

    raise NotImplementedError(
        f"coordsys '{coordsys}' not supported ({path.name}); use an XYZ array "
        f"(vla.a, vlba, ngvla-revC, wsrt, atca_6a) or supported LOC {sorted(_LOC_SITE)}."
    )
