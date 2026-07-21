"""
Observation simulator: array config + pointing (RA, Dec) + how long we observe
  ->  the uv sampling function S(u,v).

Separates "where the telescope is" (arrays.py) from "what/when we point at" (here).
Earth-rotation synthesis on a real time axis: we step through the observation in
integration-time chunks, compute the hour angle of the source at each step from
the local sidereal time (HA = LST - RA), drop the steps when the source is below
the horizon, and rotate every baseline into the (u,v) plane.
"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Angle

from arrays import antennas_local_equatorial, C


def _transit_centered_start(ra_deg, duration_h, lon_deg, ref="2026-01-01T00:00:00"):
    """UTC start time that centres a `duration_h` observation on the sources'
    transit (HA=0), where the uv coverage is symmetric and the source highest."""
    ref_t = Time(ref)
    lst = ref_t.sidereal_time("apparent", longitude=lon_deg * u.deg)
    ha_ref = Angle(lst - ra_deg * u.deg).wrap_at(12 * u.hourangle)
    dt_solar_h = (-ha_ref.hourangle) / 1.0027379  # sidereal -> solar hours to transit
    return ref_t + dt_solar_h * u.hour - (duration_h / 2) * u.hour


def observe(
    ra_deg,
    dec_deg,
    start=None,
    duration_h=6.0,
    integration_s=30.0,
    array="vla.a",
    freq=4.0e9,
    horizon_deg=8.0,
):
    """Simulate an observation and return the uv sampling.

    Parameters
    ----------
    ra_deg, dec_deg : source pointing (J2000-ish), degrees
    start           : UTC start time (ISO string); None -> centred on transit
    duration_h      : total observation length, hours
    integration_s   : dump / integration time, seconds  (sets uv-track density)
    array           : configuration name, e.g. 'vla.a', 'vlba', 'meerkat'
                      (any file in configs/; see arrays.list_arrays())
    freq            : observing frequency, Hz
    horizon_deg     : elevation limit; steps below this are discarded

    Returns
    -------
    u, v, w : coordinates in wavelengths (w kept for wide-field / w-projection;
              narrow-field imaging simply ignores it)
    info    : dict with observation diagnostics
    """
    xyz_le, lat, lon_deg = antennas_local_equatorial(array)  # local-equatorial [m]
    wl, dec = C / freq, np.deg2rad(dec_deg)
    if start is None:
        start = _transit_centered_start(ra_deg, duration_h, lon_deg)

    # --- time axis -> hour angle of the source at each dump ---
    n_dump = max(1, int(round(duration_h * 3600.0 / integration_s)))
    t = Time(start) + np.arange(n_dump) * integration_s * u.s
    lst = t.sidereal_time("apparent", longitude=lon_deg * u.deg)
    ha = Angle(lst - ra_deg * u.deg).wrap_at(12 * u.hourangle).radian

    # --- keep only the dumps when the source is above the horizon ---
    alt = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))
    up = alt > np.deg2rad(horizon_deg)
    ha_up = ha[up]

    # --- baselines (wavelengths) rotated into (u,v) at each hour angle ---
    xyz = xyz_le / wl
    i, j = np.triu_indices(len(xyz), 1)
    B = xyz[i] - xyz[j]  # (n_baseline, 3)

    cosH, sinH = np.cos(ha_up), np.sin(ha_up)
    uu = np.outer(sinH, B[:, 0]) + np.outer(cosH, B[:, 1])
    vv = (
        -np.sin(dec) * np.outer(cosH, B[:, 0])
        + np.sin(dec) * np.outer(sinH, B[:, 1])
        + np.cos(dec) * B[:, 2]
    )
    ww = (
        np.cos(dec) * np.outer(cosH, B[:, 0])
        - np.cos(dec) * np.outer(sinH, B[:, 1])
        + np.sin(dec) * B[:, 2]
    )

    info = {
        "n_baseline": len(i),
        "n_dump": n_dump,
        "n_dump_up": int(up.sum()),
        "n_vis": uu.size,
        "ha_up_hours": (ha_up.min() * 12 / np.pi, ha_up.max() * 12 / np.pi)
        if up.any()
        else None,
        "max_elev_deg": float(np.rad2deg(alt.max())),
        "w_range": (float(ww.min()), float(ww.max())) if up.any() else None,
    }
    return uu.ravel(), vv.ravel(), ww.ravel(), info
