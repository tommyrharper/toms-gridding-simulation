"""TDD for ``dirty_image_fft`` — start with one tiny step at a time, then
cross-check against the exact DFT ground truth in ``simulate.dirty_image``.

The gold-standard check below is agreement with ``dirty_image`` (a
brute-force DFT with no gridding approximation) for a *well-resolved*
synthetic uv distribution -- i.e. enough grid cells relative to the uv
extent that the convolutional-gridding approximation is accurate.
Undersampled grids (too few pixels for the baseline range) are expected to
disagree with the DFT; that is aliasing, not a bug, so it is deliberately
not exercised here.
"""

import numpy as np
import pytest

from gridding_sim.gridtools import (
    dirty_image_fft,
    least_misfit_gridder,
    spheroidal_gridder,
)
from gridding_sim.simulate import ARCSEC, dirty_image, point_source_vis


def _box_kernel(nu, W=6):
    nu = np.asarray(nu, float)
    return (np.abs(nu) < 0.5).astype(float)


def _well_resolved_uv(npix, cell, n_vis, rng, fill=0.35):
    """Random baselines filling a disk within `fill` of the grid half-width,
    so gridding has enough resolution to approximate the DFT accurately."""
    du = 1.0 / (npix * cell * ARCSEC)
    r = rng.uniform(0, npix * fill, n_vis) * du
    theta = rng.uniform(0, 2 * np.pi, n_vis)
    return r * np.cos(theta), r * np.sin(theta)


def test_returns_real_image_of_shape_npix():
    """First step: return a real (npix, npix) image, not None."""
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    img = dirty_image_fft(
        u, v, V, npix=8, cell=1.0, kernel=_box_kernel, corr_kind="spheroidal", W=6
    )
    assert img.shape == (8, 8)
    assert np.isrealobj(img)
    assert img.dtype == np.float64


def test_origin_visibility_gives_nonzero_image_after_fft():
    """Next step: actually transform the grid — don't return an unused zero array.

    One visibility at (u,v)=(0,0) fills the uv origin; its iFFT is a flat
    nonzero image (not a point-source peak). Enough to force ifftshift → ifft2
    → fftshift → .real for now; correction_map comes later.
    """
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    img = dirty_image_fft(
        u, v, V, npix=8, cell=1.0, kernel=_box_kernel, corr_kind="spheroidal", W=6
    )
    assert not np.allclose(img, 0.0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------
def test_empty_visibilities_give_zero_real_grid():
    """Regression test: the empty-input early return used to crash with
    `zeros() got an unexpected keyword argument 'dtypes'`."""
    npix = 8
    u = np.array([])
    v = np.array([])
    V = np.array([], dtype=np.complex128)
    img = dirty_image_fft(u, v, V, npix, 1.0, spheroidal_gridder, "spheroidal", W=6)
    assert img.shape == (npix, npix)
    assert img.dtype == np.float64
    assert np.allclose(img, 0.0)


def test_output_has_no_nan_or_inf():
    rng = np.random.default_rng(3)
    npix, cell = 16, 1.0
    u, v = _well_resolved_uv(npix, cell, 60, rng)
    V = point_source_vis(u, v, [(0.0, 0.0, 1.0)])
    img = dirty_image_fft(u, v, V, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    assert np.all(np.isfinite(img))


# ---------------------------------------------------------------------------
# Gold-standard cross-check: matches the exact DFT ground truth for a
# well-resolved synthetic uv distribution.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("l_arcsec,m_arcsec,flux", [(0.0, 0.0, 1.0), (3.0, -2.0, 2.5)])
def test_matches_dft_ground_truth_for_point_source(l_arcsec, m_arcsec, flux):
    rng = np.random.default_rng(1)
    npix, cell = 32, 0.5
    u, v = _well_resolved_uv(npix, cell, 800, rng)
    sources = [(l_arcsec * ARCSEC, m_arcsec * ARCSEC, flux)]
    V = point_source_vis(u, v, sources)

    img_dft = dirty_image(u, v, V, npix, cell)
    img_fft = dirty_image_fft(u, v, V, npix, cell, spheroidal_gridder, "spheroidal", W=6)

    assert np.corrcoef(img_fft.ravel(), img_dft.ravel())[0, 1] > 0.99
    assert img_fft.max() == pytest.approx(img_dft.max(), rel=0.01)
    assert np.unravel_index(np.argmax(img_fft), img_fft.shape) == np.unravel_index(
        np.argmax(img_dft), img_dft.shape
    )


def test_least_misfit_correction_also_matches_dft_ground_truth():
    rng = np.random.default_rng(1)
    npix, cell = 32, 0.5
    u, v = _well_resolved_uv(npix, cell, 800, rng)
    V = point_source_vis(u, v, [(0.0, 0.0, 1.0)])

    img_dft = dirty_image(u, v, V, npix, cell)
    img_lm = dirty_image_fft(u, v, V, npix, cell, least_misfit_gridder, "least_misfit", W=6)

    assert np.corrcoef(img_lm.ravel(), img_dft.ravel())[0, 1] > 0.99
    assert img_lm.max() == pytest.approx(img_dft.max(), rel=0.01)


# ---------------------------------------------------------------------------
# Input validation / error contract (delegated to correction_map)
# ---------------------------------------------------------------------------
def test_invalid_corr_kind_raises_value_error():
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    with pytest.raises(ValueError, match="kind must be"):
        dirty_image_fft(u, v, V, 8, 1.0, spheroidal_gridder, "not-a-real-kind", W=6)


# ---------------------------------------------------------------------------
# Linearity / superposition (the whole pipeline -- gridding, FFT, correction
# -- is linear in V, so these must hold exactly, not just approximately)
# ---------------------------------------------------------------------------
def test_scales_linearly_with_visibility_amplitude():
    rng = np.random.default_rng(2)
    npix, cell = 16, 1.0
    u, v = _well_resolved_uv(npix, cell, 60, rng)
    V1 = point_source_vis(u, v, [(0.0, 0.0, 1.0)])
    V2 = point_source_vis(u, v, [(0.0, 0.0, 3.5)])

    img1 = dirty_image_fft(u, v, V1, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    img2 = dirty_image_fft(u, v, V2, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    assert np.allclose(img2, 3.5 * img1)


def test_is_additive_over_point_sources():
    rng = np.random.default_rng(4)
    npix, cell = 16, 1.0
    u, v = _well_resolved_uv(npix, cell, 60, rng)

    sources = [(0.0, 0.0, 1.0), (2.0 * ARCSEC, -1.0 * ARCSEC, 2.0)]
    V_together = point_source_vis(u, v, sources)
    V_a = point_source_vis(u, v, sources[:1])
    V_b = point_source_vis(u, v, sources[1:])

    together = dirty_image_fft(u, v, V_together, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    part_a = dirty_image_fft(u, v, V_a, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    part_b = dirty_image_fft(u, v, V_b, npix, cell, spheroidal_gridder, "spheroidal", W=6)
    assert np.allclose(together, part_a + part_b)
