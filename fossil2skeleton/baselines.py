"""Small, transparent reconstruction baselines."""
from __future__ import annotations

import numpy as np


def mean_shape_completion(train_shapes: np.ndarray, partial: np.ndarray) -> np.ndarray:
    """Fill missing landmarks with the training mean; retain observed points."""
    train_shapes = np.asarray(train_shapes, dtype=float)
    partial = np.asarray(partial, dtype=float)
    if train_shapes.ndim != 3 or train_shapes.shape[-1] != 3:
        raise ValueError("train_shapes must have shape (samples, landmarks, 3)")
    if partial.ndim != 2 or partial.shape != train_shapes.shape[1:]:
        raise ValueError("partial must have shape (landmarks, 3)")
    mean = train_shapes.mean(axis=0)
    visible = np.isfinite(partial).all(axis=-1)
    result = mean.copy()
    result[visible] = partial[visible]
    return result.astype(np.float32)


def nearest_shape_completion(train_shapes: np.ndarray, partial: np.ndarray) -> np.ndarray:
    """Copy the training specimen closest on observed landmarks."""
    train_shapes = np.asarray(train_shapes, dtype=float)
    partial = np.asarray(partial, dtype=float)
    visible = np.isfinite(partial).all(axis=-1)
    if not visible.any():
        raise ValueError("At least one observed landmark is required.")
    distances = ((train_shapes[:, visible] - partial[visible]) ** 2).mean(axis=(1, 2))
    result = train_shapes[int(np.argmin(distances))].copy()
    result[visible] = partial[visible]
    return result.astype(np.float32)


def landmark_rmse(prediction: np.ndarray, truth: np.ndarray, scored: np.ndarray) -> float:
    """Euclidean coordinate RMSE across selected landmarks."""
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    scored = np.asarray(scored, dtype=bool)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError("prediction and truth must both have shape (landmarks, 3)")
    if scored.shape != (prediction.shape[0],) or not scored.any():
        raise ValueError("scored must select at least one landmark.")
    return float(np.sqrt(np.mean(np.sum((prediction[scored] - truth[scored]) ** 2, axis=-1))))
