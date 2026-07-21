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

def observe(ra_deg, dec_deg, start=None, duration_h=6.0, integration_s=30.0, array="vla.a", freq=4.0e9, horizon_deg=8.0):
    """Simulate an observation and return the uv sampling.
    
    Parameters
    ----------
    ra_deg, dec_deg : source pointing (J2000-ish), degrees
    start           : UTC start time (ISO string); None -> centred on transit
    duration_h      : total observation length, hours
    integration_s   : dump / integration time, seconds (sets uv-track density)
    array           : configuration name, e.g. 'vla.a', 'vlba', 'meerkat'
                      (any file in configs/; see arrays.list_arrays())
    freq            : observing frequency, Hz
    horizon_deg     : elevation limit; steps below htis are discarded
    
    Returns
    -------
    u, v, w : coordinates in wavelengths (w kept for wide-field / w-projection;
              narrow-field imaging simply ignores it)
    info    : dict with observation diagnostics
    """
    # xyz_le, lat, lon_deg = antennas_local_equatorial(array) # local-equatorial [m]

