import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pruning_gradient_privacy.cli import main
from pruning_gradient_privacy.core import (analytic_recovery, gradient_matching, invert_gradient,
    linear_gradient, nested_masks, quadratic_pruning, run_experiments, wanda_scores)


class TheoryTests(unittest.TestCase):
    def test_taylor_bound_for_indefinite_quadratics_and_groups(self):
        rng = np.random.default_rng(11)
        for _ in range(25):
            a = rng.normal(size=(5, 5))
            h = (a + a.T) / 2
            w, c = rng.normal(size=(2, 5))
            mask = rng.random(5) > 0.5
            t = rng.random()
            result = quadratic_pruning(w, h, c, mask, t)
            pruned = w - t * w * mask
            loss = lambda v: 0.5 * v @ h @ v + c @ v
            self.assertAlmostEqual(result["signed_loss_change"], loss(pruned) - loss(w))
            self.assertLessEqual(result["absolute_score_error"], result["remainder_bound"] + 1e-12)

    def test_stationary_point_needs_second_order(self):
        result = quadratic_pruning([1.0], [[1.0]], [-1.0], np.array([True]))
        self.assertEqual(result["first_order_absolute_score"], 0.0)
        self.assertEqual(result["actual_absolute_change"], 0.5)

    def test_wanda_matches_single_coordinate_output_norm(self):
        rng = np.random.default_rng(21)
        x, w = rng.normal(size=(7, 4)), rng.normal(size=(3, 4))
        scores = wanda_scores(w, x)
        for i, j in np.ndindex(w.shape):
            pruned = w.copy()
            pruned[i, j] = 0
            self.assertAlmostEqual(scores[i, j], np.linalg.norm(x @ w.T - x @ pruned.T))

    def test_wanda_ranking_can_reverse_downstream_loss_ranking(self):
        x, w = np.diag([10.0, 1.0]), np.ones((1, 2))
        score = wanda_scores(w, x)[0]
        loss = lambda v: (x @ v.T)[1, 0]
        changes = []
        for j in range(2):
            pruned = w.copy()
            pruned[0, j] = 0
            changes.append(abs(loss(pruned) - loss(w)))
        self.assertGreater(score[0], score[1])
        self.assertLess(changes[0], changes[1])


class InversionTests(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[0.6, -0.2], [0.1, 0.8]])
        self.b, self.y, self.x = np.array([0.2, -0.1]), np.array([0.3, 0.2]), np.array([0.4, -0.3])
        self.g = linear_gradient(self.w, self.b, self.x, self.y)

    def test_weight_and_bias_gradients_against_loss_finite_differences(self):
        theta = np.r_[self.w.ravel(), self.b]
        numeric = np.zeros_like(theta)
        def loss(v):
            r = v[:4].reshape(2, 2) @ self.x + v[4:] - self.y
            return 0.5 * r @ r
        for index in range(theta.size):
            delta = np.eye(theta.size)[index] * 1e-6
            numeric[index] = (loss(theta + delta) - loss(theta - delta)) / 2e-6
        np.testing.assert_allclose(self.g, numeric, atol=1e-10)

    def test_gradient_matching_derivative_under_partial_observation(self):
        z = np.array([-0.2, 0.5])
        mask = np.array([True, False, True, True, False, True])
        _, derivative = gradient_matching(self.w, self.b, z, self.y, self.g, mask)
        numeric = np.zeros_like(z)
        for j in range(z.size):
            delta = np.eye(z.size)[j] * 1e-6
            plus, _ = gradient_matching(self.w, self.b, z + delta, self.y, self.g, mask)
            minus, _ = gradient_matching(self.w, self.b, z - delta, self.y, self.g, mask)
            numeric[j] = (plus - minus) / 2e-6
        np.testing.assert_allclose(derivative, numeric, rtol=1e-7, atol=1e-10)

    def test_masks_are_nested_and_ties_use_lowest_index(self):
        for strategy in ["taylor", "magnitude", "random"]:
            masks = nested_masks(np.ones(8), np.ones(8), [0.1, 0.5, 1], strategy, 123)
            self.assertTrue(np.all(masks[0] <= masks[1]))
            self.assertTrue(np.all(masks[1] <= masks[2]))
            if strategy != "random":
                np.testing.assert_array_equal(np.flatnonzero(masks[1]), np.arange(4))
            else:
                repeated = nested_masks(np.ones(8), np.ones(8), [0.5], strategy, 123)
                np.testing.assert_array_equal(masks[1], repeated[0])

    def test_full_gradient_analytic_recovery_and_zero_gradient_failure(self):
        np.testing.assert_allclose(analytic_recovery(self.g, 2, 2), self.x, atol=1e-14)
        with self.assertRaises(ValueError):
            analytic_recovery(np.zeros(6), 2, 2)

    def test_no_observation_baseline_performs_no_updates(self):
        mask = nested_masks(np.r_[self.w.ravel(), self.b], self.g, [0], "taylor")[0]
        initial = np.array([-0.3, 0.1])
        self.assertFalse(mask.any())
        result = invert_gradient(self.w, self.b, self.y, self.g, mask, initial, 100)
        np.testing.assert_array_equal(result["reconstruction"], initial)
        self.assertEqual(result["optimizer_updates"], 0)

    def test_optimizer_decreases_observed_error_without_true_input(self):
        mask = np.ones(self.g.size, dtype=bool)
        initial = np.array([-0.3, 0.1])
        before = gradient_matching(self.w, self.b, initial, self.y, self.g, mask)[0]
        result = invert_gradient(self.w, self.b, self.y, self.g, mask, initial, 250)
        self.assertLess(result["matching_objective"], before * 1e-5)
        np.testing.assert_allclose(result["reconstruction"], self.x, atol=1e-4)
        zero = invert_gradient(self.w, self.b, self.y, self.g, mask, initial, 0)
        np.testing.assert_array_equal(zero["reconstruction"], initial)

    def test_reproducible_suite_and_finite_results(self):
        a = run_experiments(seed=7, dimension=2, trials=1, rhos=[0.5, 1], budgets=[0, 10])
        b = run_experiments(seed=7, dimension=2, trials=1, rhos=[0.5, 1], budgets=[0, 10])
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        json.dumps(a, allow_nan=False)
        self.assertEqual(len(a["inversion"]), 12)
        self.assertFalse(a["scope"]["epiplexity_estimated"])
        for budget in [0, 10]:
            full = [r["matching_objective"] for r in a["inversion"] if r["rho"] == 1 and r["budget"] == budget]
            self.assertEqual(min(full), max(full))


class ValidationAndCliTests(unittest.TestCase):
    def test_invalid_numeric_parameters(self):
        for args in [{"seed": -1}, {"dimension": 0}, {"trials": 0}, {"budgets": [-1]},
                     {"budgets": []}, {"rhos": [-0.1]}, {"rhos": [float("nan")]}, {"rhos": [1.1]}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_experiments(**args)
        with self.assertRaises(ValueError):
            wanda_scores([[float("inf")]], [[1]])
        with self.assertRaises(ValueError):
            quadratic_pruning([1, 2], [[1, 2], [0, 1]], [0, 0], np.ones(2, dtype=bool))

    def test_cli_artifacts_are_machine_readable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            argv = ["run", "--output", str(path), "--trials", "1", "--dimension", "2", "--budgets", "0,5"]
            self.assertEqual(main(argv), 0)
            first = {p.name: p.read_bytes() for p in path.iterdir()}
            self.assertEqual(set(first), {"results.json", "inversion.csv", "report.md", "reconstruction.svg"})
            self.assertEqual(main(argv), 0)
            self.assertEqual(first, {p.name: p.read_bytes() for p in path.iterdir()})
            self.assertEqual(json.loads(first["results.json"])["scope"]["budget_unit"], "Adam optimizer updates")


if __name__ == "__main__":
    unittest.main()
