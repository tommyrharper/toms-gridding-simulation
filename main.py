import numpy as np
import matplotlib.pyplot as plt

from arrays import list_arrays
from observe import observe
from simulate import ARCSEC, dirty_beam, w_term_error

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

"""
Constants for plotting the UV coverage and the dirty beam
"""
radio_array = RADIO[0]
# Right ascension degrees - celestial longitude (like longitude on Earth)
ra_deg = 0.0
# Declination degrees - celestial latitude (-90° to + 90°)
dec_deg = 34.0
# Observation length - 5 minutes for a near-snapshot, or 1 hour. Longer tracks fill in the uv plane via earth rotation.
duration_h = 1.0  # hours.  5 min = 5/60 ≈ 0.083 ;  1 hour = 1.0

"""
Constants for the imaging grid and w-term check
"""
# image size [npix*npix]
npix = 256
# pixel size [arcsec] (VLA-A resolution ~0.4" at 4 GHz)
cell = 0.10


def plot_uv_coverage_and_dirty_beam(u, v, info, npix=192, show_plot=False):
    if u.size == 0:  # source never rises for this array
        print(
            f"'{radio_array}' never sees Dec {dec_deg:.0f} deg above the horizon "
            f"(max elevation {info['max_elev_deg']:.1f} deg). Try another Dec / array."
        )
        return
    bmax = np.hypot(u, v).max()
    cell = (1.0 / bmax) / ARCSEC / 3.0  # ~3 px across the beam
    step = max(1, u.size // 80000)  # thin big arrays for the DFT
    beam = dirty_beam(u[::step], v[::step], npix, cell)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(u, v, s=1, alpha=0.4)
    ax[0].scatter(-u, -v, s=1, alpha=0.4)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"uv coverage — {radio_array}")
    ax[0].set_xlabel(r"u [$\lambda$]")
    ax[0].set_ylabel(r"v [$\lambda$]")
    ax[1].imshow(beam.T, origin="lower", cmap="cubehelix", vmin=-0.05, vmax=0.3)
    ax[1].set_title(f'dirty beam  (cell = {cell:.3f}")')
    print(f"n_vis = {info['n_vis']},  max elev = {info['max_elev_deg']:.1f} deg")
    if show_plot:
        plt.show()


def check_narrow_field_approximation(w):
    dphi = w_term_error(w, npix, cell)
    print(
        f'array = {radio_array}, FoV = {npix * cell:.1f}", |w|max = {np.abs(w).max():3e} lambda'
    )
    if dphi < 0.1:
        print(" -> negligible: safe to drop w (narrow-field OK)")
    elif dphi < 1.0:
        print(" -> marginal: fine near the centre, errors grow toward the edge")
    else:
        print(" -> w MATTERS: shrink npix*cell, or use w-projection")


def main():
    print(len(list_arrays()), "configurations available")
    print("Current antenna array: ", radio_array)
    print("Right ascension degrees: ", ra_deg)
    print("Declination degrees: ", dec_deg)
    print("Duration hours: ", duration_h)

    observations = observe(ra_deg, dec_deg, duration_h=duration_h, array=radio_array)
    u, v, w, info = observations
    plot_uv_coverage_and_dirty_beam(u, v, info, show_plot=False)
    check_narrow_field_approximation(w)


if __name__ == "__main__":
    main()
