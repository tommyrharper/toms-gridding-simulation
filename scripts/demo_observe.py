import numpy as np

import _repo_path  # noqa: F401  — put src/ on path before package imports
from demo_config import DEMO

from gridding_sim.arrays import list_arrays
from gridding_sim.observe import observe
from gridding_sim.simulate import (
    dirty_image,
    field_halfwidth_arcsec,
    make_point_sources,
    point_source_vis,
    make_dirty_beam,
)
from gridding_sim.diagnostics import (
    check_narrow_field_approximation,
    require_visibilities,
    fft_residuals,
    print_residual_stats,
)
from gridding_sim.plotting import plot_demo_summary
from gridding_sim.gridtools import spheroidal_gridder, least_misfit_gridder
from gridding_sim.imaging import dirty_image_fft


def main() -> None:
    cfg = DEMO

    print(len(list_arrays()), "configurations available")
    print("Current antenna array: ", cfg.radio_array)
    print("Right ascension degrees: ", cfg.ra_deg)
    print("Declination degrees: ", cfg.dec_deg)
    print("Duration hours: ", cfg.duration_h)

    u, v, w, info = observe(
        cfg.ra_deg, cfg.dec_deg, duration_h=cfg.duration_h, array=cfg.radio_array
    )
    require_visibilities(u, info, cfg.radio_array, cfg.dec_deg)
    print(f"n_vis = {info['n_vis']},  max elev = {info['max_elev_deg']:.1f} deg")
    check_narrow_field_approximation(w, cfg.radio_array, cfg.npix, cfg.cell)

    hw = field_halfwidth_arcsec(cfg.npix, cfg.cell)
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
    print(f"sky_mode={cfg.sky_mode!r}: {len(sources)} source(s), {info['n_vis']} visibilities")

    img_dft = dirty_image(u, v, V, cfg.npix, cfg.cell)
    img_sph = dirty_image_fft(u, v, V, cfg.npix, cfg.cell, spheroidal_gridder, "spheroidal")
    img_lm = dirty_image_fft(u, v, V, cfg.npix, cfg.cell, least_misfit_gridder, "least_misfit")
    print("DFT peak flux:", img_dft.max())

    beam, beam_cell = make_dirty_beam(u, v)
    d_sph, d_lm, vmax, inner = fft_residuals(img_dft, img_sph, img_lm)
    print_residual_stats({"spheroidal": d_sph, "least-misfit": d_lm}, inner)

    plot_demo_summary(
        u,
        v,
        cfg.radio_array,
        beam,
        beam_cell,
        img_dft,
        d_sph,
        d_lm,
        vmax,
        show_plot=cfg.show_plot,
    )


if __name__ == "__main__":
    main()
