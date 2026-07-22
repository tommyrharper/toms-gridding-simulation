"""Matplotlib helpers for visualising uv coverage, the dirty beam, and dirty images."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def draw_uv_coverage(
    ax: Axes,
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    array: str,
) -> None:
    ax.scatter(u, v, s=1, alpha=0.4)
    ax.scatter(-u, -v, s=1, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"uv coverage — {array}")
    ax.set_xlabel(r"u [$\lambda$]")
    ax.set_ylabel(r"v [$\lambda$]")


def draw_dirty_beam(ax: Axes, beam: npt.NDArray[np.float64], cell: float) -> None:
    ax.imshow(beam.T, origin="lower", cmap="cubehelix", vmin=-0.05, vmax=0.3)
    ax.set_title(f'dirty beam  (cell = {cell:.3f}")')


def draw_image(
    ax: Axes,
    img: npt.NDArray[np.float64],
    title: str,
    *,
    cmap: str = "cubehelix",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    fig: Figure | None = None,
) -> None:
    im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if colorbar:
        if fig is None:
            fig = ax.figure
        fig.colorbar(im, ax=ax, shrink=0.8)


def make_demo_summary_axes(fig: Figure) -> dict[str, Axes]:
    """Top: uv | beam. Bottom: DFT | spheroidal residual | least-misfit residual."""
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.45)
    return {
        "uv": fig.add_subplot(gs[0, 0:3]),
        "beam": fig.add_subplot(gs[0, 3:6]),
        "dft": fig.add_subplot(gs[1, 0:2]),
        "sph": fig.add_subplot(gs[1, 2:4]),
        "lm": fig.add_subplot(gs[1, 4:6]),
    }


def plot_demo_summary(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    array: str,
    beam: npt.NDArray[np.float64],
    beam_cell: float,
    img_dft: npt.NDArray[np.float64],
    d_sph: npt.NDArray[np.float64],
    d_lm: npt.NDArray[np.float64],
    vmax: float,
    *,
    show_plot: bool = True,
) -> Figure:
    """One figure: uv + beam (top), DFT + FFT residuals (bottom). Draw-only."""
    fig = plt.figure(figsize=(15, 8.2))
    ax = make_demo_summary_axes(fig)

    draw_uv_coverage(ax["uv"], u, v, array)
    draw_dirty_beam(ax["beam"], beam, beam_cell)
    draw_image(ax["dft"], img_dft, "DFT (ground truth)", colorbar=True, fig=fig)
    draw_image(
        ax["sph"],
        d_sph,
        "DFT − spheroidal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        colorbar=True,
        fig=fig,
    )
    draw_image(
        ax["lm"],
        d_lm,
        "DFT − least-misfit",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        colorbar=True,
        fig=fig,
    )

    if show_plot:
        plt.show()
    return fig
