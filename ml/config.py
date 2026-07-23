"""Tunable parameters for the ML baseline (mirrors scripts/demo_config.py)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLConfig:
    # Observation used to generate a single training example
    radio_array: str = "vla.d"
    ra_deg: float = 0.0
    dec_deg: float = 34.0
    duration_h: float = 0.05
    freq: float = 4.0e9

    # Sky model (reuses gridding_sim.simulate.make_point_sources)
    sky_mode: str = "single"
    n_sources: int = 3
    flux: float = 2.0

    # Imaging grid -- kept small: a plain MLP's output is npix*npix with no
    # convolutional structure, so this stays far below demo_config's npix=256.
    npix: int = 32
    cell: float = 1.0  # arcsec / pixel

    # Visibility capacity (padding/truncation target -- see ml/data.py)
    max_vis: int = 4096

    # Model
    hidden_sizes: tuple[int, ...] = (256, 256)

    # Reproducibility
    seed: int = 0


MLCONFIG = MLConfig()
