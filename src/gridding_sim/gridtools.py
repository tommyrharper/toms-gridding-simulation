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
    """Deposit visibilities onto a regular uv grid by convolutional gridding.

    Parameters
    ----------
    u, v, V : array_like
        Visibility coordinates [wavelengths] and complex visibilities.
    npix : int
        Grid size (npix x npix).
    cell : float
        Image-plane pixel size [arcsec]. Sets FoV = npix * cell and
        uv-cell size du = 1 / FoV_rad.
    kernel : callable
        1-D gridding kernel ``kernel(nu, W=W)`` in units of uv-cells,
        zero outside ``|nu| < W/2`` (see ``spheroidal_gridder``).
    W : int
        Kernel support in uv-cells (passed through to ``kernel``).

    Returns
    -------
    grid : ndarray, complex128, shape (npix, npix)
        Accumulated gridded visibilities. Index ``i`` corresponds to
        continuous coordinate ``(i - npix // 2) * du`` (image-centred,
        matching ``simulate.dirty_image``). No normalisation — that
        belongs in ``dirty_image_fft``.

    Contract (for each visibility k)::

        du = 1 / (npix * cell * ARCSEC)
        for i, j in 0..npix-1:
            nu = u[k]/du - (i - npix//2)
            nv = v[k]/du - (j - npix//2)
            grid[i, j] += V[k] * kernel(nu, W=W) * kernel(nv, W=W)

        Skip (i, j) outside the array (no periodic wrap).
    """
    # Pseudocode for grid_visibilities:
    #
    # 1. Create a zeroed complex grid of shape (npix, npix).
    # 2. If u, v, V are empty: return zero grid.
    # 3. Compute du = 1 / (npix * cell * ARCSEC)  # uv cell size in lambda units.
    # 4. For each visibility:
    #     - Find the centre position (uc, vc) in pixel units: uc = u[k]/du + npix//2, etc.
    #     - For i in range of grid pixels within kernel support W of uc:
    #         - For j in range within W of vc:
    #             - Compute nu, nv (distance in uv cell units).
    #             - If i, j in-bounds:
    #                 - grid[i, j] += V[k] * kernel(nu, W) * kernel(nv, W)
    # 5. Return the grid.

    G = np.zeros((npix, npix), dtype=np.complex128)
    return G
    raise NotImplementedError("implement convolutional gridding (TDD)")


def dirty_image_fft(u, v, V, npix, cell, kernel, corr_kind, W=6):
    print('u', u)
    grid = grid_visibilities(u, v, V, npix, cell, kernel, W)
    return
