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
