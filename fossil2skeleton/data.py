"""Synthetic landmark data and mask utilities for the Fossil2Skeleton demo.

The generator is deliberately a toy statistical world, not a biological model.
"""
from __future__ import annotations

import numpy as np


def make_synthetic_shapes(
    n_samples: int = 512,
    n_landmarks: int = 24,
    latent_dim: int = 6,
    seed: int = 7,
) -> np.ndarray:
    """Return (samples, landmarks, xyz) toy shapes from a low-rank shape space.

    Coordinates combine an elliptical reference shape with shared latent
    deformation modes and small noise. No real specimen data are used.
    """
    if n_samples < 2 or n_landmarks < 4 or latent_dim < 1:
        raise ValueError("Need >=2 samples, >=4 landmarks, and >=1 latent factor.")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_landmarks, endpoint=False)
    template = np.stack((np.cos(t), 0.65 * np.sin(t), 0.3 * np.sin(2 * t)), axis=1)
    modes = rng.normal(0, 0.12, size=(latent_dim, n_landmarks, 3))
    modes -= modes.mean(axis=1, keepdims=True)
    factors = rng.normal(size=(n_samples, latent_dim))
    shapes = template + np.einsum("sk,knc->snc", factors, modes)
    shapes += rng.normal(0, 0.008, size=shapes.shape)
    return shapes.astype(np.float32)


def random_observation_mask(
    n_samples: int,
    n_landmarks: int,
    observed_fraction: float = 0.5,
    seed: int = 11,
) -> np.ndarray:
    """Boolean (samples, landmarks) mask; True means observed/preserved."""
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be strictly between 0 and 1.")
    if n_samples < 1 or n_landmarks < 2:
        raise ValueError("Need at least one sample and two landmarks.")
    rng = np.random.default_rng(seed)
    mask = rng.random((n_samples, n_landmarks)) < observed_fraction
    # Ensure each specimen retains at least one observation and one missing point.
    for row in mask:
        if not row.any():
            row[rng.integers(n_landmarks)] = True
        if row.all():
            row[rng.integers(n_landmarks)] = False
    return mask


def apply_mask(shapes: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Hide missing coordinates using NaN; preserve visible landmarks."""
    shapes = np.asarray(shapes)
    observed = np.asarray(observed, dtype=bool)
    if shapes.ndim != 3 or shapes.shape[-1] != 3:
        raise ValueError("shapes must have shape (samples, landmarks, 3)")
    if observed.shape != shapes.shape[:2]:
        raise ValueError("mask must have shape (samples, landmarks)")
    return np.where(observed[..., None], shapes, np.nan).astype(np.float32)
