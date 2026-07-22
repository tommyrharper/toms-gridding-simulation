"""
Given helpers for the gridding exercise (Step 9).  You do NOT need to modify this
file -- call these from your own gridding code.

Two gridding kernels (both zero outside |nu| < W/2) and their image-plane
corrections are provided:

    spheroidal_gridder(nu, W=6)        the prolate-spheroidal kernel  (closed form)
    least_misfit_gridder(nu, W=6)      the least-misfit kernel        (W=6, x0=0.25)
    correction_map(npix, kind, W=6)    1-D image correction h  (apply as img*outer(h,h))
                                       kind = "spheroidal" | "least_misfit"

Derivation: Ye, Gull, Tan & Nikolic, "Optimal gridding and degridding in radio
interferometry imaging", MNRAS 2019 (arXiv:1906.07102).  The least-misfit kernel
is the paper's optimum; the spheroidal is the classical choice -- compare them!
"""
import numpy as np
from scipy.special import pro_ang1

from .simulate import ARCSEC

# ---------------------------------------------------------------------------
# spheroidal gridding function (closed form)
# ---------------------------------------------------------------------------
def spheroidal_gridder(nu, W=6):
    """0th-order prolate spheroidal gridding function, zero outside |nu| < W/2"""
    nu = np.asarray(nu, float)
    t = 2.0 * nu / W
    out = np.zeros_like(nu)
    g = t * t < 1.0
    out[g] = np.sqrt(1.0 - t[g] ** 2) * pro_ang1(1, 1, W / 2 * np.pi, t[g])[0]
    return out


# ---------------------------------------------------------------------------
# WIP: grid + FFT dirty image (compare to the DFT ground truth in simulate.py)
# ---------------------------------------------------------------------------
def grid_visibilities(u, v, V, npix, cell, kernel, W=6):
    fov = npix * cell * ARCSEC

    return


def dirty_image_fft(u, v, V, npix, cell, kernel, corr_kind, W=6):
    print('u', u)
    grid = grid_visibilities(u, v, V, npix, cell, kernel, W)
    return
