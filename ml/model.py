"""A very simple MLP baseline: padded visibilities -> flattened dirty image."""
import torch
from torch import nn

from .data import N_FEATURES


class VisMLP(nn.Module):
    """Flatten (max_vis, N_FEATURES) -> Linear+ReLU stack -> flattened (npix*npix,)."""

    def __init__(
        self,
        max_vis: int,
        npix: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        in_features = max_vis * N_FEATURES
        out_features = npix * npix

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)
        self.max_vis = max_vis
        self.npix = npix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, max_vis, N_FEATURES) -> (batch, npix*npix), both float32."""
        return self.net(x.reshape(x.shape[0], -1))
