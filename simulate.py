"""
Ideal, fully-analytic visibility + dirty-image simulator for the gridding ML testbed.

Everything here is EXACT, no gridding approximation anywhere:
  - a point source is a delta in the image  ->  its visibility is an analytic fringe
  - the sampling function S(u,v) is just the set of uv points we choose to measure
  - the dirty image is a brute-force DFT (the T1 "slow map")

Textbook picture:
    V_measured(u,v) = V_true(u,v) * S(u,v)
    dirty_image     = true_sky (*) dirty_beam,   dirty_beam = FT of S(u,v)

Conventions match the Imaging-Tutorial (Ye et al. 2019):
  u,v in wavelengths;  vis = sum_k S_k exp(+2 pi i (u l + v m));
  dirty image  I(x,y) = (1/sum w) Re sum_k w_k V_k exp(+2 pi i (u_k x + v_k y)).
Narrow field: w-term ignored (n ~ 1).
"""

import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)  # radians per arcsec


# ---------------------------------------------------------------------------
# 3. Exact dirty image  (brute-force DFT)  and dirty beam
# ---------------------------------------------------------------------------
def dirty_image(u, v, V, npix, cell_arcsec, weights=None):
    """Exact DFT dirty image, shape (npix, npix). cell in arcsec, image centred."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    if weights is None:
        weights = np.ones(u.shape)
    cell = cell_arcsec * ARCSEC
    x = (np.arange(npix) - npix // 2) * cell  # pixel coords [rad]
    pu = np.exp(2j * np.pi * np.outer(u, x))  # (n_vis, npix)
    pv = np.exp(2j * np.pi * np.outer(v, x))  # (n_vis, npix)
    M = (weights * V)[:, None] * pu  # (n_vis, npix)
    img = (M.T @ pv).real / weights.sum()  # (npix, npix)
    return img


def dirty_beam(u, v, npix, cell_arcsec, weights=None):
    """Dirty beam = dirty image of a unit point source at the phase centre."""
    ones = np.ones(np.asarray(u).shape, dtype=np.complex128)
    return dirty_image(u, v, ones, npix, cell_arcsec, weights)


def w_term_error(w, npix, cell_arcsec):
    """Peak narrow-field phase error [rad] from dropping w, at the field edge:
    |dphi| = 2 pi |w|max (1 - cos theta_edge). << 1 rad => w is safe to ingore."""
    theta_edge = (npix // 2) * cell_arcsec * ARCSEC
    return 2 * np.pi * np.abs(w).max() * (1.0 - np.cos(theta_edge))


# ---------------------------------------------------------------------------
# 1. Sky  ->  analytic visibility function  V_true(u,v)
# ---------------------------------------------------------------------------
def field_halfwidth_arcsec(npix, cell_arcsec, margin=0.8):
    """Max source offset from the centre (arcsec) that stays safely inside the image."""
    return margin * (npix // 2) * cell_arcsec
