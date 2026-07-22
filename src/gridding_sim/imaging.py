"""Approximate dirty-image pipeline: convolutional gridding + iFFT.

This is the classical forward path under study (compare to the exact DFT
ground truth in ``simulate.dirty_image``). Kernels live in ``gridtools``.
"""

from typing import Callable
import math

import numpy as np
import numpy.typing as npt

from .gridtools import correction_map
from .simulate import ARCSEC


def grid_visibilities(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    V: npt.NDArray[np.complexfloating],
    npix: int,
    cell: float,
    kernel: Callable[..., npt.NDArray[np.float64]],
    W: float = 6,
) -> npt.NDArray[np.complex128]:
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

    # Convert inputs ot NumPy arrays so indexing behaves consistently
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    V = np.asarray(V, dtype=np.complex128)

    if u.ndim != 1 or v.ndim != 1 or V.ndim != 1:
        raise ValueError("u, v, and V must be one-dimensional arrays")

    if not (len(u) == len(v) == len(V)):
        raise ValueError("u, v, and V must have the same length")

    if npix <= 0:
        raise ValueError("npix must be positive")

    if cell <= 0:
        raise ValueError("cell must be positive")

    if W <= 0:
        raise ValueError("W must be positive")

    # Create the empty regular uv grid
    G = np.zeros((npix, npix), dtype=np.complex128)

    # Empty inputs produce an empty grid
    if len(V) == 0:
        return G

    pixel_size_radians = cell * ARCSEC
    # image width is also known as the FoV or field of view
    fov = npix * pixel_size_radians
    # but we are in inverse space
    du = 1.0 / fov
    # get centre element (index representing u = 0, v =0)
    c = npix // 2
    # half-wdith of the kernel support in grid-cell units
    half_width = W / 2.0

    # process each visibility sample
    for k, vis in enumerate(V):
        u_coord = u[k]
        v_coord = v[k]

        # covert physical uv coordinates into continuous array-index coordinates
        uc = (u_coord / du) + c
        vc = (v_coord / du) + c

        # construct conservative integer bounds around the visibility
        i_start = math.floor(uc - half_width)
        i_stop = math.ceil(uc + half_width)

        j_start = math.floor(vc - half_width)
        j_stop = math.ceil(vc + half_width)

        # for each visibility, visit nearby regular uv-grid points
        for i in range(i_start, i_stop + 1):
            if i < 0 or i >= npix:
                continue

            # signed distance from point i to the visibility
            nu = uc - i

            # Kernel is zero outside its support
            if abs(nu) >= half_width:
                continue

            # get kernel weight for this dist from centre (force a scalar: a
            # kernel returning a non-0-d array crashes the `G[i, j] +=` below)
            weight_u = float(np.asarray(kernel(nu, W=W)).item())

            for j in range(j_start, j_stop + 1):
                if j < 0 or j >= npix:
                    continue
                # signed distance from point j to the visibility
                nv = vc - j

                # distance must be within the kernel
                if abs(nv) >= half_width:
                    continue

                # get kernel weight for this dist from centre
                weight_v = float(np.asarray(kernel(nv, W=W)).item())

                # convolved visibility pixel
                convolved_visibility_pixel = vis * weight_u * weight_v

                # deposit this visibility onto the regular grid
                G[i, j] += convolved_visibility_pixel

    return G


def dirty_image_fft(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    V: npt.NDArray[np.complexfloating],
    npix: int,
    cell: float,
    kernel: Callable[..., npt.NDArray[np.float64]],
    corr_kind: str,
    W: float = 6,
) -> npt.NDArray[np.float64]:
    V = np.asarray(V, dtype=np.complex128)

    if len(V) == 0:
        return np.zeros((npix, npix), dtype=np.float64)

    grid = grid_visibilities(u, v, V, npix, cell, kernel, W)

    """
    Demonstrate what np.fft.ifftshift does:
    Original array: [0 1 2 3 4 5 6 7]
    After ifftshift: [4 5 6 7 0 1 2 3]
    Original 2D array:
    [[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    [12 13 14 15]]
    After ifftshift (2D):
    [[10 11  8  9]
    [14 15 12 13]
    [ 2  3  0  1]
    [ 6  7  4  5]]
    """
    # Convert from centred uv ordering to NumPy FFT ordering.
    uv_grid_in_fft_order = np.fft.ifftshift(grid)

    # Inverse Fourier transform into the image plane.
    dirty_image_in_fft_order = np.fft.ifft2(uv_grid_in_fft_order)

    # Put l = 0, m = 0 at the centre of the image.
    dirty_image_centred = np.fft.fftshift(dirty_image_in_fft_order)

    # this corrects for a scaling factor in ifft2
    # ifft2 scales by 1/N^2 (N = number of pixel)
    # exact DFT scales by the number of visibilities
    normalisation = npix**2 / len(V)

    # Normalise the image
    normalised_dirty_image_centred = dirty_image_centred * normalisation

    # apply the gridding correction
    h = correction_map(
        npix=npix,
        kind=corr_kind,
        W=W,
    )

    normalised_dirty_image_centred *= np.outer(h, h)

    return normalised_dirty_image_centred.real
