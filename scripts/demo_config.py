"""Tunable parameters shared by scripts/demo_observe.py and scripts/demo_app.py.

Edit the values on DEMO below to change the demo observation — array,
pointing, integration time, imaging grid, and sky model — without touching
the driver script itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

RADIO: list[str] = [
    "vla.a",  # 0
    "vla.b",  # 1
    "vla.c",  # 2
    "vla.d",  # 3
    "vlba",  # 4
    "ngvla-revC",  # 5
    "wsrt",  # 6
    "atca_6a",  # 7
    "meerkat",  # 8
]


@dataclass(frozen=True)
class DemoConfig:
    # Observation
    radio_array: str = RADIO[0]
    ra_deg: float = 0.0  # right ascension [deg]
    dec_deg: float = 34.0  # declination [deg]
    duration_h: float = 1.0  # hours (5 min ≈ 0.083)

    # Imaging grid (DFT dirty image / w-term check)
    npix: int = 256
    cell: float = 0.10  # arcsec / pixel

    # Sky model — "single" | "random" | "manual"
    sky_mode: str = "single"
    n_sources: int = 5  # used when sky_mode == "random"
    flux: float = 2.0  # Jy, used when sky_mode == "single" (random draws from a fixed 0.5-5.0 Jy range instead)
    manual_sources: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 2.0), (5.0, -3.0, 1.0)]
    )  # (x_arcsec, y_arcsec, flux_Jy), used when sky_mode == "manual"
    rng_seed: int = 0

    # Plotting
    show_plot: bool = True


DEMO = DemoConfig()
