"""Optional PyTorch transformer for masked 3D landmark completion.

This module describes a research prototype. It has not been trained or tested
on biological specimens. Uncertainty is estimated by a stochastic ensemble,
not calibrated as a probability of anatomical correctness.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # give a useful error while keeping baseline utilities usable
    raise ImportError("The transformer requires PyTorch. Install requirements.txt.") from exc


class LandmarkTransformer(nn.Module):
    """Predict coordinates for every landmark from coordinates plus an observed mask."""

    def __init__(
        self,
        n_landmarks: int = 24,
        width: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.n_landmarks = n_landmarks
        self.coord_projection = nn.Linear(3, width)
        self.mask_embedding = nn.Embedding(2, width)
        self.position_embedding = nn.Embedding(n_landmarks, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, coordinates, observed):
        """coordinates: (B,N,3), observed: (B,N) bool; missing coords are zero."""
        if coordinates.ndim != 3 or coordinates.shape[1:] != (self.n_landmarks, 3):
            raise ValueError(f"coordinates must have shape (B,{self.n_landmarks},3)")
        if observed.shape != coordinates.shape[:2]:
            raise ValueError("observed must have shape (B,N)")
        safe_coords = torch.where(observed[..., None], coordinates, torch.zeros_like(coordinates))
        positions = torch.arange(self.n_landmarks, device=coordinates.device)
        h = self.coord_projection(safe_coords)
        h = h + self.mask_embedding(observed.long()) + self.position_embedding(positions)[None]
        return self.output(self.encoder(h))


def complete_with_uncertainty(model, coordinates, observed, draws: int = 16):
    """MC-dropout completion: mean and per-landmark coordinate standard deviation."""
    if draws < 2:
        raise ValueError("draws must be >=2 to estimate spread")
    was_training = model.training
    model.train()  # keep dropout active for stochastic forward passes
    with torch.no_grad():
        samples = torch.stack([model(coordinates, observed) for _ in range(draws)])
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)
        mean = torch.where(observed[..., None], coordinates, mean)
        std = torch.where(observed[..., None], torch.zeros_like(std), std)
    model.train(was_training)
    return mean, std
