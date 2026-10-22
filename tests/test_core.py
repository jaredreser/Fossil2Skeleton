import unittest
import numpy as np

from fossil2skeleton.baselines import (
    landmark_rmse, mean_shape_completion, nearest_shape_completion,
)
from fossil2skeleton.data import apply_mask, make_synthetic_shapes, random_observation_mask


class SyntheticDataTests(unittest.TestCase):
    def test_generator_is_reproducible_and_shaped(self):
        a = make_synthetic_shapes(12, 10, 3, seed=4)
        b = make_synthetic_shapes(12, 10, 3, seed=4)
        self.assertEqual(a.shape, (12, 10, 3))
        np.testing.assert_array_equal(a, b)

    def test_masks_retain_both_observed_and_missing_points(self):
        mask = random_observation_mask(30, 20, .4, seed=2)
        self.assertEqual(mask.shape, (30, 20))
        self.assertTrue(mask.any(axis=1).all())
        self.assertTrue((~mask).any(axis=1).all())

    def test_rejects_invalid_observed_fraction(self):
        for fraction in (0.0, 1.0, -0.2, 1.2):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                random_observation_mask(2, 5, fraction)

    def test_apply_mask_preserves_visible_and_hides_other_points(self):
        shapes = make_synthetic_shapes(3, 8)
        mask = random_observation_mask(3, 8, .5)
        partial = apply_mask(shapes, mask)
        np.testing.assert_array_equal(partial[mask], shapes[mask])
        self.assertTrue(np.isnan(partial[~mask]).all())


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[2, 2, 2], [3, 2, 2], [2, 3, 2], [2, 2, 3]],
        ], dtype=np.float32)
        self.truth = self.train[0].copy()
        self.partial = self.truth.copy()
        self.partial[1:] = np.nan
        self.missing = np.array([False, True, True, True])

    def test_mean_shape_fills_only_missing_landmarks(self):
        result = mean_shape_completion(self.train, self.partial)
        np.testing.assert_array_equal(result[0], self.truth[0])
        np.testing.assert_allclose(result[1], [2, 1, 1])

    def test_nearest_shape_uses_observed_geometry_and_preserves_observations(self):
        result = nearest_shape_completion(self.train, self.partial)
        np.testing.assert_array_equal(result, self.truth)

    def test_rmse_scores_only_selected_landmarks(self):
        pred = self.truth.copy()
        pred[0] = 100  # should not affect the score
        self.assertEqual(landmark_rmse(pred, self.truth, self.missing), 0.0)
        pred[1, 0] += 2
        self.assertAlmostEqual(landmark_rmse(pred, self.truth, self.missing), np.sqrt(4 / 3))

    def test_rejects_no_visible_landmarks(self):
        with self.assertRaises(ValueError):
            nearest_shape_completion(self.train, np.full((4, 3), np.nan))

    def test_rejects_wrong_landmark_mask_shape(self):
        with self.assertRaises(ValueError):
            mean_shape_completion(self.train, np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
