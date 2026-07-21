import numpy as np
import matplotlib.pyplot as plt

from arrays import list_arrays
from observe import observe
from simulate import (
    ARCSEC,
    dirty_beam,
    dirty_image,
    w_term_error,
    field_halfwidth_arcsec,
    make_point_sources,
    point_source_vis,
)

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

# Observation
radio_array = RADIO[0]
ra_deg = 0.0  # right ascension [deg]
dec_deg = 34.0  # declination [deg]
duration_h = 1.0  # hours (5 min ≈ 0.083)

# Imaging grid (DFT dirty image / w-term check)
npix = 256
cell = 0.10  # arcsec / pixel


def require_visibilities(u, info, array, dec):
    if u.size == 0:
        raise ValueError(
            f"'{array}' never sees Dec {dec:.0f}° above the horizon "
            f"(max elev {info['max_elev_deg']:.1f}°). Try another Dec / array."
        )


def beam_cell_arcsec(u, v):
    """Pixel size with ~3 px across the main lobe (PSF diagnostic zoom)."""
    bmax = np.hypot(u, v).max()
    return (1.0 / bmax) / ARCSEC / 3.0


def plot_uv_coverage_and_dirty_beam(u, v, info, array, dec, npix=192, show_plot=False):
    """Uv coverage + dirty beam on a beam-matched grid (not the imaging FoV)."""
    require_visibilities(u, info, array, dec)

    cell = beam_cell_arcsec(u, v)
    step = max(1, u.size // 80000)  # thin big arrays for the DFT
    beam = dirty_beam(u[::step], v[::step], npix, cell)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(u, v, s=1, alpha=0.4)
    ax[0].scatter(-u, -v, s=1, alpha=0.4)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"uv coverage — {array}")
    ax[0].set_xlabel(r"u [$\lambda$]")
    ax[0].set_ylabel(r"v [$\lambda$]")
    ax[1].imshow(beam.T, origin="lower", cmap="cubehelix", vmin=-0.05, vmax=0.3)
    ax[1].set_title(f'dirty beam  (cell = {cell:.3f}")')
    print(f"n_vis = {info['n_vis']},  max elev = {info['max_elev_deg']:.1f} deg")
    if show_plot:
        plt.show()


def plot_dft_dirty_image(img, show_plot=False):
    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(img.T, origin="lower", cmap="cubehelix")
    plt.title("DFT dirty image (ground truth)")
    plt.colorbar()
    plt.show()


def check_narrow_field_approximation(w, array, npix, cell):
    dphi = w_term_error(w, npix, cell)
    print(
        f'array = {array}, FoV = {npix * cell:.1f}", |w|max = {np.abs(w).max():3e} lambda'
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

    u, v, w, info = observe(ra_deg, dec_deg, duration_h=duration_h, array=radio_array)
    plot_uv_coverage_and_dirty_beam(u, v, info, radio_array, dec_deg, show_plot=False)
    check_narrow_field_approximation(w, radio_array, npix, cell)

    hw = field_halfwidth_arcsec(npix, cell)
    print(f'manual coordinate range: {-hw:+.2f}" .. {hw:+.2f}" on each axis')

    sky_mode = "single"  # "single" | "random" | "manual"
    sources = make_point_sources(
        sky_mode,
        npix,
        cell,
        n=5,
        flux=2.0,
        manual=[(0.0, 0.0, 2.0), (5.0, -3.0, 1.0)],
        rng=np.random.default_rng(0),
    )
    V = point_source_vis(u, v, sources)
    print(f"sky_mode={sky_mode!r}: {len(sources)} source(s), {info['n_vis']} visibilities")

    img_dft = dirty_image(u, v, V, npix, cell)
    print("peak flux:", img_dft.max())
    plot_dft_dirty_image(img_dft, show_plot=True)


if __name__ == "__main__":
    main()
