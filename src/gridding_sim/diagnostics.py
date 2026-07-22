"""Sanity checks / diagnostic prints for an observation and its imaging setup."""

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


def check_narrow_field_approximation(
    w: npt.NDArray[np.float64], array: str, npix: int, cell: float
) -> None:
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


def print_residual_stats(
    residuals: dict[str, npt.NDArray[np.float64]],
    inner: slice,
) -> None:
    for name, d in residuals.items():
        e = d[inner, inner]
        print(
            f"{name:13s}: inner-field error  "
            f"max={np.abs(e).max():.2e}  rms={np.sqrt((e**2).mean()):.2e}"
        )
