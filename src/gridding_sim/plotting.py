"""Matplotlib helpers for visualising uv coverage, the dirty beam, and dirty images."""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .observe import ObserveInfo
from .simulate import ARCSEC, dirty_beam
from .diagnostics import require_visibilities


def beam_cell_arcsec(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float:
    """Pixel size with ~3 px across the main lobe (PSF diagnostic zoom)."""
    bmax = np.hypot(u, v).max()
    return (1.0 / bmax) / ARCSEC / 3.0


def plot_uv_coverage_and_dirty_beam(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    info: ObserveInfo,
    array: str,
    dec: float,
    npix: int = 192,
    show_plot: bool = False,
) -> None:
    """Uv coverage + dirty beam on a beam-matched grid (not the imaging FoV)."""
    require_visibilities(u, info, array, dec)

    cell = beam_cell_arcsec(u, v)
    step = max(1, u.size // 80000)  # thin big arrays for the DFT
    beam = dirty_beam(u[::step], v[::step], npix, cell)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(u, v, s=1, alpha=0.4)
    ax[0].scatter(-u, -v, s=1, alpha=0.4)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"uv coverage — {array}")
    ax[0].set_xlabel(r"u [$\lambda$]")
    ax[0].set_ylabel(r"v [$\lambda$]")
    ax[1].imshow(beam.T, origin="lower", cmap="cubehelix", vmin=-0.05, vmax=0.3)
    ax[1].set_title(f'dirty beam  (cell = {cell:.3f}")')
    print(f"n_vis = {info['n_vis']},  max elev = {info['max_elev_deg']:.1f} deg")
    if show_plot:
        plt.show()


def plot_dft_dirty_image(
    img: npt.NDArray[np.float64], show_plot: bool = False
) -> None:
    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(img.T, origin="lower", cmap="cubehelix")
    plt.title("DFT dirty image (ground truth)")
    plt.colorbar()
    if show_plot:
        plt.show()
