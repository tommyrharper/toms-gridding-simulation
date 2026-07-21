import numpy as np
import matplotlib.pyplot as plt

from arrays import list_arrays
from observe import observe

# ─────────────────────────────────────────────────────────────────────────────
#  Antenna configurations in configs/  (241 total; run list_arrays() for them all).
#  Pick one by its file stem, e.g.  array = "vla.a".   Resolution ∝ 1 / max-baseline.
# ─────────────────────────────────────────────────────────────────────────────
#  VLA — Very Large Array, New Mexico, 27×25 m, cm (1–50 GHz), reconfigurable "Y":
#     vla.a   max baseline 36 km   highest res (~0.04–1")  compact / point sources
#     vla.b   max baseline 11 km   high res
#     vla.c   max baseline 3.4 km  moderate res
#     vla.d   max baseline 1.0 km  lowest res — extended emission, large-scale flux
#     vla.bna vla.cnb vla.dnc      hybrid (extended North arm) — low / southern Dec
#  VLBA — Very Long Baseline Array, 10 antennas across the continent:
#     vlba    ~8600 km baselines   milli-arcsec res — AGN cores, masers, astrometry
#  ngVLA — next-generation VLA (planned, use for forecasts):
#     ngvla-revC / -revD           full array
#     ngvla-core/-plains/-mid/-main/-lba/-sba/-gb-vlba   sub-arrays
#  ALMA — Atacama mm/submm, 50×12 m, Chile:
#     alma.cycleN.M                N = cycle, M = 1 (compact ~0.16 km) … 10 (~16 km)
#     aca.* , aca.cycle*           Atacama Compact Array (7 m) — large-scale mm flux
#  Other radio:
#     meerkat  64×13.5 m, South Africa, L/UHF/S — SKA precursor; HI, transients
#     wsrt     Westerbork, Netherlands — E–W 14×25 m
#     atca_*   ATCA, Australia, 6×22 m — many E–W / hybrid (atca_6a, atca_750a, atca_ew352 …)
#     sma.*    Submillimeter Array, Mauna Kea (submm)
#     carma.*  CARMA (decommissioned) mm ;   pdbi-a…d  Plateau de Bure / NOEMA mm
#  Supported coord systems: XYZ (VLA/VLBA/ngVLA/WSRT/ATCA) + LOC (ALMA/ACA/MeerKAT/SMA).
#  UTM configs (a few ALMA / CARMA) are NOT supported.

RADIO = [
    "vla.a",        # 0
    "vla.b",        # 1
    "vla.c",        # 2
    "vla.d",        # 3
    "vlba",         # 4
    "ngvla-revC",   # 5
    "wsrt",         # 6
    "atca_6a",      # 7
    "meerkat",      # 8
]

radio_array = RADIO[0]
# Right ascension degrees - celestial longitude (like longitude on Earth)
ra_deg = 0.0
# Declination degrees - celestial latitude (-90° to + 90°)
dec_deg = 34.0
# Observation length - 5 minutes for a near-snapshot, or 1 hour. Longer tracks fill in the uv plane via earth rotation.
duration_h = 1.0 # hours.  5 min = 5/60 ≈ 0.083 ;  1 hour = 1.0

def show_uv_beam(array, ra, dec, duration_h, npix=192):
    u, v, w, info = observe(ra, dec, duration_h=duration_h, array=array)

def main():
    print(len(list_arrays()), "configurations available")

    print("Current antenna array: ", radio_array)
    print("Right ascension degrees: ", ra_deg)
    print("Declination degrees: ", dec_deg)
    print("Duration hours: ", duration_h)


if __name__ == "__main__":
    main()
