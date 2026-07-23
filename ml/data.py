"""Pack raw (u, v, V) visibilities into fixed-size tensors for VisMLP.

Deliberately minimal, foundation-only:
  - No feature normalisation. u, v are raw wavelength-scale values (can be
    ~1e4-1e5 for long-baseline arrays) fed straight into the model. Fine for
    a wiring/shape smoke test, not for real training -- scaling belongs in a
    follow-up once training actually starts.
  - No Dataset/DataLoader. `make_example` builds one (input, target) pair;
    batching several examples is just `torch.stack(...)` /
    `x.unsqueeze(0)` until more than one example is needed.
"""
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch

from gridding_sim.simulate import PointSource, dirty_image, point_source_vis

N_FEATURES = 5  # (u, v, Re(V), Im(V), mask)


def pad_visibilities(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    V: npt.NDArray[np.complexfloating],
    max_vis: int,
) -> torch.Tensor:
    """Pack raw (u, v, V) into a fixed (max_vis, 5) float32 tensor.

    Columns: (u, v, Re(V), Im(V), mask). u, v in wavelengths, V in Jy.
    mask = 1.0 for a real sample, 0.0 for padding.

    Truncation: if more than `max_vis` samples are given, only the first
    `max_vis` are kept (deterministic -- shuffle beforehand for random
    subsampling). Padding: remaining rows are zero, including mask.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    V = np.asarray(V, dtype=np.complex128)

    if max_vis <= 0:
        raise ValueError("max_vis must be positive")

    n = min(len(u), max_vis)
    out = np.zeros((max_vis, N_FEATURES), dtype=np.float32)
    out[:n, 0] = u[:n]
    out[:n, 1] = v[:n]
    out[:n, 2] = V[:n].real
    out[:n, 3] = V[:n].imag
    out[:n, 4] = 1.0

    return torch.from_numpy(out)


def make_example(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    sources: Sequence[PointSource],
    max_vis: int,
    npix: int,
    cell_arcsec: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one (input, target) training pair.

    input  : float32, shape (max_vis, N_FEATURES) -- see `pad_visibilities`.
    target : float32, shape (npix*npix,) -- flattened `simulate.dirty_image`
             ground truth (reshape with `.view(npix, npix)` to visualise).
    """
    V = point_source_vis(u, v, sources)
    x = pad_visibilities(u, v, V, max_vis)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    n = min(len(u_arr), max_vis)
    img = dirty_image(u_arr[:n], v_arr[:n], V[:n], npix, cell_arcsec)
    y = torch.from_numpy(np.asarray(img, dtype=np.float32).ravel())
    return x, y
