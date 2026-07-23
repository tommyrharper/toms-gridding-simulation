"""Smoke test for the ML baseline: one forward pass, one backward pass, one
optimizer step. Proves the model/data wiring works end-to-end -- not a
training loop (no epochs, no CLI args, no plotting)."""
import numpy as np
import torch
import torch.nn.functional as F

import _repo_path  # noqa: F401  — put src/ and repo root on path before imports
from ml.config import MLCONFIG
from ml.data import make_example
from ml.model import VisMLP

from gridding_sim.observe import observe
from gridding_sim.simulate import make_point_sources
from gridding_sim.diagnostics import require_visibilities


def main() -> None:
    cfg = MLCONFIG
    torch.manual_seed(cfg.seed)

    u, v, w, info = observe(
        cfg.ra_deg,
        cfg.dec_deg,
        duration_h=cfg.duration_h,
        array=cfg.radio_array,
        freq=cfg.freq,
    )
    require_visibilities(u, info, cfg.radio_array, cfg.dec_deg)
    print(f"n_vis = {info['n_vis']}  (capacity max_vis = {cfg.max_vis})")
    if info["n_vis"] > cfg.max_vis:
        print(f"warning: truncating {info['n_vis']} visibilities to max_vis={cfg.max_vis}")

    sources = make_point_sources(
        cfg.sky_mode,
        cfg.npix,
        cfg.cell,
        n=cfg.n_sources,
        flux=cfg.flux,
        rng=np.random.default_rng(cfg.seed),
    )
    x, y = make_example(u, v, sources, cfg.max_vis, cfg.npix, cfg.cell)

    model = VisMLP(cfg.max_vis, cfg.npix, cfg.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pred = model(x.unsqueeze(0))
    loss = F.mse_loss(pred, y.unsqueeze(0))
    print(f"initial loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("backward + optimizer step OK -- wiring verified")


if __name__ == "__main__":
    main()
