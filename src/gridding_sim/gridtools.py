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
import numpy.typing as npt
from scipy.special import pro_ang1


# ---------------------------------------------------------------------------
# spheroidal gridding function (closed form)
# ---------------------------------------------------------------------------
def spheroidal_gridder(nu: npt.ArrayLike, W: float = 6) -> npt.NDArray[np.float64]:
    """0th-order prolate spheroidal gridding function, zero outside |nu| < W/2"""
    nu = np.asarray(nu, float)
    t = 2.0 * nu / W
    out = np.zeros_like(nu)
    g = t * t < 1.0
    out[g] = np.sqrt(1.0 - t[g] ** 2) * pro_ang1(1, 1, W / 2 * np.pi, t[g])[0]
    return out


# ---------------------------------------------------------------------------
# least-misfit gridding function  (optimised correction h for W=6, x0=0.25)
# ---------------------------------------------------------------------------
_X0 = 0.25
_OPT6_H = np.array([
    1.0001930023325103, 1.0007722487834718, 1.0017384582470565, 1.0030928306497886,
    1.0048370496094119, 1.006973286170906,  1.0095042036335624, 1.0124329634778302,
    1.0157632324212278, 1.0194991906233397, 1.0236455410561094, 1.0282075201008483,
    1.033190909366273,  1.0386020488054259, 1.0444478511548989, 1.0507358177532391,
    1.0574740557831654, 1.0646712970185641, 1.0723369181167883, 1.0804809625436269,
    1.0891141642026927, 1.0982479728515449, 1.1078945814002472, 1.1180669551825217,
    1.1287788633171774, 1.1400449122631175, 1.1518805816930127, 1.164302262836624,
    1.1773272994022188, 1.1909740312694896, 1.2052618410867355, 1.220211203987587,
    1.2358437405714804, 1.2521822734177863, 1.2692508872950534, 1.2870749933531256,
    1.3056813975329964, 1.3250983734891064, 1.345355740288839,  1.3664849452934122,
    1.3885191524603193, 1.4114933365321043, 1.4354443834948476, 1.4604111977049021,
    1.4864348162134051, 1.513558530761332,  1.5418280180050101, 1.5712914785751284,
    1.6019997856026427, 1.6340066433994047, 1.6673687570562927, 1.7021460137842996,
    1.7384016768275126, 1.7762025929639036, 1.8156194145769096, 1.8567268374316457,
    1.8996038553867987, 1.9443340333039465, 1.9910057996541108, 2.0397127602994884,
    2.0905540351680947, 2.1436346196866345, 2.1990657728474763, 2.2569654342222729])


def _calc_gridder_as_C(h, x0, nu, W):
    """Least-misfit gridder taps C(l-nu) from the correction h (tutorial T2)."""
    factor = 1.0 / np.sqrt(2.0)
    nu = np.atleast_1d(np.asarray(nu, float))
    M, N = len(nu), len(h)
    C = np.zeros((W, M))
    x = x0 * np.arange(N + 1, dtype=float) / N
    h_ext = np.concatenate(([1.0], h))
    rhs = np.r_[np.ones(N + 1), np.zeros(N + 1)]
    rhs[0] *= factor; rhs[N] *= factor
    B = np.zeros((2 * N + 2, W))
    for mi, nv in enumerate(nu):
        for r in range(W):
            k = r - (W / 2) + 1
            B[:N + 1, r] = h_ext * np.cos(2 * np.pi * (k - nv) * x)
            B[N + 1:, r] = h_ext * np.sin(2 * np.pi * (k - nv) * x)
        B[0, :] *= factor; B[N, :] *= factor; B[N + 1, :] *= factor; B[2 * N + 1, :] *= factor
        C[:, mi] = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return C


_LM_PROFILE = None


def _lm_profile(W=6, oversample=4096):
    """Build (and cache) the least-misfit kernel profile C(eta) over [-W/2, W/2]."""
    global _LM_PROFILE
    if _LM_PROFILE is None:
        f = (np.arange(oversample) + 0.5) / oversample          # fractional offsets
        C = _calc_gridder_as_C(_OPT6_H, _X0, f, W)              # (W, oversample)
        taps = np.arange(-(W // 2) + 1, W // 2 + 1)             # m values [-2..3]
        eta = (taps[:, None] - f[None, :]).ravel()             # C sampled at eta = m - f
        o = np.argsort(eta)
        _LM_PROFILE = (eta[o], C.ravel()[o])
    return _LM_PROFILE


def least_misfit_gridder(nu, W=6):
    """Least-misfit gridding function (W=6, x0=0.25), zero outside |nu| < W/2."""
    if W != 6:
        raise ValueError("least_misfit_gridder only has an optimised kernel for W=6")
    nu = np.asarray(nu, float)
    eta, val = _lm_profile(W)
    out = np.interp(nu, eta, val, left=0.0, right=0.0)
    return np.where(np.abs(nu) >= W / 2, 0.0, out)



# ---------------------------------------------------------------------------
# image-plane grid correction  (works for either kernel)
# ---------------------------------------------------------------------------
def _make_evaluation_grids(W, M, N):
    nu = (np.arange(W * M, dtype=float) + 0.5) / (2 * M)
    x = np.arange(N + 1, dtype=float) / (2 * N)
    return nu, x


def _gridder_to_C(gridder, W):
    M = len(gridder) // W
    C = np.zeros((W, M))
    for r in range(W):
        l = r - (W / 2) + 1
        indx = (np.arange(M) - 2 * M * l).astype(int)
        indx[indx < 0] = -indx[indx < 0] - 1
        C[r, :] = gridder[indx]
    return C


def _grid_correction(gridder, nu, x, W):
    M = len(nu) // W
    dnu = nu[1] - nu[0]
    C = _gridder_to_C(gridder, W)
    c = np.zeros(x.shape)
    d = np.zeros(x.shape)
    for n, xv in enumerate(x):
        for rp in range(W):
            lp = rp - (W / 2) + 1
            for r in range(W):
                l = r - (W / 2) + 1
                d[n] += np.sum(C[rp, :] * C[r, :] * np.cos(2 * np.pi * (lp - l) * xv)) * dnu
            c[n] += np.sum(C[rp, :] * np.cos(2 * np.pi * (lp - nu[:M]) * xv)) * dnu
    return c / d


def correction_map(npix, kind="spheroidal", W=6, M=32):
    """1-D image-plane correction h(x) for the chosen kernel.  Apply as img*outer(h,h)."""
    nu, x = _make_evaluation_grids(W, M, npix // 2)
    if kind == "spheroidal":
        gridder = spheroidal_gridder(nu, W)
    elif kind == "least_misfit":
        gridder = least_misfit_gridder(nu, W)
    else:
        raise ValueError(f"kind must be 'spheroidal' or 'least_misfit', got {kind!r}")
    gc = _grid_correction(gridder, nu, x, W)
    h = np.zeros(npix)
    h[npix // 2:] = gc[:npix // 2]
    h[:npix // 2] = gc[:0:-1]
    return h
