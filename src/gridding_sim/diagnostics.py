"""Sanity checks / diagnostics for an observation and its imaging setup.

Each check has a print-based variant for CLI use and a plain data-returning
variant (`narrow_field_verdict`, `residual_stats`) for callers, like the
Streamlit app, that need structured results instead of parsed stdout.
"""

import numpy as np
import numpy.typing as npt

from .observe import ObserveInfo
from .simulate import w_term_error


def require_visibilities(
    u: npt.NDArray[np.float64], info: ObserveInfo, array: str, dec: float
) -> None:
    if u.size == 0:
        raise ValueError(
            f"'{array}' never sees Dec {dec:.0f}° above the horizon "
            f"(max elev {info['max_elev_deg']:.1f}°). Try another Dec / array."
        )


def narrow_field_verdict(
    w: npt.NDArray[np.float64], npix: int, cell: float
) -> tuple[float, str]:
    """Peak narrow-field phase error and a human verdict on whether w is safe to drop."""
    dphi = w_term_error(w, npix, cell)
    if dphi < 0.1:
        msg = "negligible: safe to drop w (narrow-field OK)"
    elif dphi < 1.0:
        msg = "marginal: fine near the centre, errors grow toward the edge"
    else:
        msg = "w MATTERS: shrink npix*cell, or use w-projection"
    return dphi, msg


def check_narrow_field_approximation(
    w: npt.NDArray[np.float64], array: str, npix: int, cell: float
) -> None:
    dphi, msg = narrow_field_verdict(w, npix, cell)
    print(
        f'array = {array}, FoV = {npix * cell:.1f}", |w|max = {np.abs(w).max():3e} lambda'
    )
    print(f" -> {msg}")


def fft_residuals(
    img_dft: npt.NDArray[np.float64],
    img_sph: npt.NDArray[np.float64],
    img_lm: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, slice]:
    """DFT minus each FFT image, plus a shared residual colour scale."""
    npix = img_dft.shape[0]
    inner = slice(npix // 4, 3 * npix // 4)
    d_sph = img_dft - img_sph
    d_lm = img_dft - img_lm
    vmax = float(np.abs(np.concatenate([d_sph[inner, inner], d_lm[inner, inner]])).max())
    return d_sph, d_lm, vmax, inner


def residual_stats(
    residuals: dict[str, npt.NDArray[np.float64]],
    inner: slice,
) -> dict[str, dict[str, float]]:
    """Inner-field max/rms error for each named residual image."""
    out: dict[str, dict[str, float]] = {}
    for name, d in residuals.items():
        e = d[inner, inner]
        out[name] = {"max": float(np.abs(e).max()), "rms": float(np.sqrt((e**2).mean()))}
    return out


def print_residual_stats(
    residuals: dict[str, npt.NDArray[np.float64]],
    inner: slice,
) -> None:
    for name, s in residual_stats(residuals, inner).items():
        print(f"{name:13s}: inner-field error  max={s['max']:.2e}  rms={s['rms']:.2e}")
