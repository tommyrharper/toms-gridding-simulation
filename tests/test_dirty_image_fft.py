"""TDD for ``dirty_image_fft`` — start with one tiny step at a time."""

import numpy as np

from gridding_sim.gridtools import dirty_image_fft


def _box_kernel(nu, W=6):
    nu = np.asarray(nu, float)
    return (np.abs(nu) < 0.5).astype(float)


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
