"""Shared demo pipeline: DemoConfig -> uv sampling -> DFT/FFT dirty images.

Used by both the CLI (demo_observe.py) and the interactive Streamlit app
(demo_app.py) so the two stay in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from demo_config import DemoConfig

from gridding_sim.observe import ObserveInfo, observe
from gridding_sim.simulate import (
    PointSource,
    dirty_image,
    field_halfwidth_arcsec,
    make_point_sources,
    make_dirty_beam,
    point_source_vis,
)
from gridding_sim.diagnostics import (
    check_narrow_field_approximation,
    fft_residuals,
    narrow_field_verdict,
    print_residual_stats,
    require_visibilities,
    residual_stats as compute_residual_stats,
)
from gridding_sim.gridtools import least_misfit_gridder, spheroidal_gridder
from gridding_sim.imaging import dirty_image_fft


@dataclass
class DemoResult:
    u: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    w: npt.NDArray[np.float64]
    info: ObserveInfo
    sources: list[PointSource]
    img_dft: npt.NDArray[np.float64]
    img_sph: npt.NDArray[np.float64]
    img_lm: npt.NDArray[np.float64]
    beam: npt.NDArray[np.float64]
    beam_cell: float
    d_sph: npt.NDArray[np.float64]
    d_lm: npt.NDArray[np.float64]
    vmax: float
    inner: slice
    dphi: float
    narrow_field_msg: str
    residual_stats: dict[str, dict[str, float]]


def run_demo(cfg: DemoConfig, *, verbose: bool = True) -> DemoResult:
    """Run the full observe -> sky -> DFT/FFT dirty-image pipeline for `cfg`."""
    if verbose:
        print("Current antenna array: ", cfg.radio_array)
        print("Right ascension degrees: ", cfg.ra_deg)
        print("Declination degrees: ", cfg.dec_deg)
        print("Duration hours: ", cfg.duration_h)

    u, v, w, info = observe(
        cfg.ra_deg, cfg.dec_deg, duration_h=cfg.duration_h, array=cfg.radio_array
    )
    require_visibilities(u, info, cfg.radio_array, cfg.dec_deg)
    if verbose:
        print(f"n_vis = {info['n_vis']},  max elev = {info['max_elev_deg']:.1f} deg")
        check_narrow_field_approximation(w, cfg.radio_array, cfg.npix, cfg.cell)
    dphi, narrow_field_msg = narrow_field_verdict(w, cfg.npix, cfg.cell)

    hw = field_halfwidth_arcsec(cfg.npix, cfg.cell)
    if verbose:
        print(f'manual coordinate range: {-hw:+.2f}" .. {hw:+.2f}" on each axis')

    sources = make_point_sources(
        cfg.sky_mode,
        cfg.npix,
        cfg.cell,
        n=cfg.n_sources,
        flux=cfg.flux,
        manual=cfg.manual_sources,
        rng=np.random.default_rng(cfg.rng_seed),
    )
    V = point_source_vis(u, v, sources)
    if verbose:
        print(f"sky_mode={cfg.sky_mode!r}: {len(sources)} source(s), {info['n_vis']} visibilities")

    # Exact perfect dirty image
    img_dft = dirty_image(u, v, V, cfg.npix, cfg.cell)
    # Gridded iFFT imperfect dirty images
    img_sph = dirty_image_fft(u, v, V, cfg.npix, cfg.cell, spheroidal_gridder, "spheroidal")
    img_lm = dirty_image_fft(u, v, V, cfg.npix, cfg.cell, least_misfit_gridder, "least_misfit")
    if verbose:
        print("DFT peak flux:", img_dft.max())

    beam, beam_cell = make_dirty_beam(u, v)
    # how accurate is the gridding + iFFT image
    d_sph, d_lm, vmax, inner = fft_residuals(img_dft, img_sph, img_lm)
    stats = compute_residual_stats({"spheroidal": d_sph, "least-misfit": d_lm}, inner)
    if verbose:
        print_residual_stats({"spheroidal": d_sph, "least-misfit": d_lm}, inner)

    return DemoResult(
        u=u,
        v=v,
        w=w,
        info=info,
        sources=sources,
        img_dft=img_dft,
        img_sph=img_sph,
        img_lm=img_lm,
        beam=beam,
        beam_cell=beam_cell,
        d_sph=d_sph,
        d_lm=d_lm,
        vmax=vmax,
        inner=inner,
        dphi=dphi,
        narrow_field_msg=narrow_field_msg,
        residual_stats=stats,
    )
