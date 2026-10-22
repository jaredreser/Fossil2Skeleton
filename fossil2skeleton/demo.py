"""Run a dependency-light baseline demonstration on synthetic landmark shapes."""
from __future__ import annotations

import numpy as np

from .baselines import landmark_rmse, mean_shape_completion, nearest_shape_completion
from .data import make_synthetic_shapes, random_observation_mask


def main():
    shapes = make_synthetic_shapes(n_samples=128, seed=7)
    masks = random_observation_mask(128, shapes.shape[1], observed_fraction=0.55, seed=9)
    train = shapes[:96]
    truth = shapes[100]
    observed = masks[100]
    partial = np.where(observed[:, None], truth, np.nan)
    missing = ~observed
    mean = mean_shape_completion(train, partial)
    nearest = nearest_shape_completion(train, partial)
    print("Synthetic held-out specimen. RMSE on hidden landmarks only:")
    print(f"  training mean : {landmark_rmse(mean, truth, missing):.4f}")
    print(f"  nearest shape : {landmark_rmse(nearest, truth, missing):.4f}")
    print("These toy scores do not measure performance on real anatomy.")


if __name__ == "__main__":
    main()
