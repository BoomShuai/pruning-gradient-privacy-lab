"""Mathematics for the lab, independent of the filesystem and CLI.

Linear regression uses L = ||W x + b - y||² / 2 for one synthetic record.
The observation concatenates row-major weight gradients and bias gradients.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import numpy as np


def _array(value, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-D array")
    return array


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _mask(mask, size: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (size,):
        raise ValueError(f"mask must contain {size} booleans")
    return array


def quadratic_pruning(weights, hessian, linear, mask, scale: float = 1.0) -> dict:
    """Exact loss change, absolute first-order score, and spectral remainder bound.

    `mask=True` denotes a coordinate being pruned; scale=1 removes it fully.
    Symmetric indefinite Hessians are allowed; beta is their spectral norm.
    """
    w = _array(weights, 1, "weights")
    h = _array(hessian, 2, "hessian")
    c = _array(linear, 1, "linear")
    if h.shape != (w.size, w.size) or c.shape != w.shape:
        raise ValueError("quadratic dimensions do not match")
    if not np.allclose(h, h.T, rtol=1e-12, atol=1e-12):
        raise ValueError("hessian must be symmetric")
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("scale must be finite and in [0, 1]")
    u = float(scale) * w * _mask(mask, w.size)
    first = float(-(h @ w + c) @ u)
    remainder = float(0.5 * u @ h @ u)
    beta = float(np.max(np.abs(np.linalg.eigvalsh(h))))
    return {
        "signed_loss_change": first + remainder,
        "actual_absolute_change": abs(first + remainder),
        "first_order_absolute_score": abs(first),
        "signed_taylor_remainder": remainder,
        "absolute_score_error": abs(abs(first + remainder) - abs(first)),
        "remainder_bound": float(0.5 * beta * (u @ u)),
        "gradient_lipschitz_beta": beta,
    }


def wanda_scores(weights, activations) -> np.ndarray:
    """Single-weight output perturbation norms for Z = X W^T.

    This is not an equality with an arbitrary downstream loss change.
    """
    w = _array(weights, 2, "weights")
    x = _array(activations, 2, "activations")
    if w.shape[1] != x.shape[1]:
        raise ValueError("weights and activations must share input dimension")
    return np.abs(w) * np.linalg.norm(x, axis=0)[None, :]


def _linear_inputs(weights, bias, inputs, target):
    w = _array(weights, 2, "weights")
    b, x, y = (_array(v, 1, n) for v, n in
               [(bias, "bias"), (inputs, "inputs"), (target, "target")])
    if w.shape != (b.size, x.size) or b.shape != y.shape:
        raise ValueError("linear regression dimensions do not match")
    return w, b, x, y


def linear_gradient(weights, bias, inputs, target) -> np.ndarray:
    """Weight and bias gradients of one-record squared-error regression."""
    w, b, x, y = _linear_inputs(weights, bias, inputs, target)
    residual = w @ x + b - y
    return np.concatenate((np.outer(residual, x).ravel(), residual))


def analytic_recovery(observation, outputs: int, dimension: int) -> np.ndarray:
    """Full-gradient control: x = G_W^T G_b / ||G_b||² if G_b != 0.

    Only valid for full, unquantized weight+bias gradients from one record.
    """
    m = _integer(outputs, "outputs", 1)
    d = _integer(dimension, "dimension", 1)
    g = _array(observation, 1, "observation")
    if g.size != m * (d + 1):
        raise ValueError("full observation has the wrong size")
    gw, gb = g[:m * d].reshape(m, d), g[m * d:]
    norm2 = float(gb @ gb)
    if norm2 <= np.finfo(float).tiny:
        raise ValueError("zero bias gradient cannot identify the input")
    return gw.T @ gb / norm2


def nested_masks(weights, observation, rhos: Sequence[float], strategy: str,
                 seed: int = 42) -> list[np.ndarray]:
    """Prefixes of one fixed ordering; stable index tie breaking.

    For Taylor scores `weights` must be flattened W followed by b. The random
    baseline uses a single seeded permutation, independent of the gradient.
    """
    theta = _array(weights, 1, "weights")
    g = _array(observation, 1, "observation")
    if theta.shape != g.shape:
        raise ValueError("weights and observation must have equal shape")
    seed = _integer(seed, "seed")
    rhos = list(rhos)
    if not rhos or any(not np.isfinite(r) or not 0 <= r <= 1 for r in rhos):
        raise ValueError("rhos must be finite and in [0, 1]")
    if strategy == "taylor":
        scores = np.abs(theta * g)
        order = np.lexsort((np.arange(g.size), -scores))
    elif strategy == "magnitude":
        order = np.lexsort((np.arange(g.size), -np.abs(g)))
    elif strategy == "random":
        order = np.random.default_rng(seed).permutation(g.size)
    else:
        raise ValueError("strategy must be taylor, magnitude, or random")
    result = []
    for rho in rhos:
        mask = np.zeros(g.size, dtype=bool)
        mask[order[:math.ceil(rho * g.size)]] = True
        result.append(mask)
    return result


def gradient_matching(weights, bias, candidate, target, observation, mask):
    """Mean squared observed-gradient error / 2 and its analytic input gradient.

    The observed support is held fixed during optimization. Constraints implied
    by a data-dependent top-k selection rule are not part of this objective.
    """
    w, b, z, y = _linear_inputs(weights, bias, candidate, target)
    observed = _array(observation, 1, "observation")
    if observed.size != w.size + b.size:
        raise ValueError("observation has the wrong size")
    selected = _mask(mask, observed.size)
    count = int(selected.sum())
    if count == 0:
        # The empty observation imposes no gradient constraint. Its zero
        # objective does not indicate successful recovery; retain the prior.
        return 0.0, np.zeros_like(z)
    r = w @ z + b - y
    error = (np.concatenate((np.outer(r, z).ravel(), r)) - observed) * selected
    ew = error[:w.size].reshape(w.shape)
    eb = error[w.size:]
    derivative = (w.T @ (ew @ z + eb) + ew.T @ r) / count
    return float(0.5 * (error @ error) / count), derivative


def invert_gradient(weights, bias, target, observation, mask, initial,
                    budget: int = 100, learning_rate: float = 0.05) -> dict:
    """Adam matching from one fixed start; return best objective seen so far.

    `budget` is optimizer updates, not a FLOP or wall-clock budget. No true input
    is used by the optimizer. There is no guarantee of finding a global optimum.
    """
    budget = _integer(budget, "budget")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    z = _array(initial, 1, "initial").copy()
    value, derivative = gradient_matching(weights, bias, z, target, observation, mask)
    best, best_value = z.copy(), value
    if not np.any(mask):
        return {"reconstruction": best, "matching_objective": 0.0,
                "optimizer_updates": 0}
    first, second = np.zeros_like(z), np.zeros_like(z)
    for step in range(1, budget + 1):
        first = 0.9 * first + 0.1 * derivative
        second = 0.999 * second + 0.001 * derivative ** 2
        z -= learning_rate * (first / (1 - 0.9 ** step)) / (
            np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
        value, derivative = gradient_matching(weights, bias, z, target, observation, mask)
        if not np.isfinite(value) or not np.isfinite(derivative).all():
            raise ValueError("optimizer diverged; reduce learning_rate")
        if value < best_value:
            best, best_value = z.copy(), value
    return {"reconstruction": best, "matching_objective": best_value,
            "optimizer_updates": budget}


def _theory_examples(rng) -> dict:
    h = np.array([[2.0, 0.4], [0.4, 1.0]])
    w = np.array([1.0, -0.7])
    mask = np.array([True, True])
    taylor = [dict(scale=t, **quadratic_pruning(w, h, np.array([0.2, 0.1]), mask, t))
              for t in [0.01, 0.1, 0.5, 1.0]]
    stationary = quadratic_pruning(w, h, -h @ w, mask)
    x, weights = rng.normal(size=(5, 3)), rng.normal(size=(2, 3))
    scores = wanda_scores(weights, x)
    maximum_error = 0.0
    for i, j in np.ndindex(weights.shape):
        pruned = weights.copy()
        pruned[i, j] = 0
        actual = np.linalg.norm(x @ weights.T - x @ pruned.T)
        maximum_error = max(maximum_error, abs(float(actual - scores[i, j])))
    # Same output row: Wanda ranks feature 0 first, but a linear downstream
    # loss F(Z)=Z[1,0] responds only to feature 1, reversing both true rankings.
    return {"quadratic_taylor": taylor, "stationary_point": stationary,
            "wanda_max_output_norm_error": maximum_error,
            "wanda_loss_ranking_counterexample": {
                "activations": [[10.0, 0.0], [0.0, 1.0]], "weights": [[1.0, 1.0]],
                "downstream_loss": "F(Z) = Z[1, 0]",
                "wanda_scores": [10.0, 1.0], "true_absolute_loss_changes": [0.0, 1.0]},
            "wanda_stationary_counterexample": {
                "loss": "0.5 * (w*x - 1)^2, w=x=1",
                "taylor_score": 0.0, "wanda_score": 1.0, "true_loss_change": 0.5}}


def run_experiments(seed: int = 42, dimension: int = 6, trials: int = 3,
                    rhos=(0.0, 0.25, 0.5, 0.75, 1.0), budgets=(0, 25, 100)) -> dict:
    """Deterministic synthetic suite. Identical starts across strategies/rhos."""
    seed = _integer(seed, "seed")
    dimension, trials = _integer(dimension, "dimension", 1), _integer(trials, "trials", 1)
    budgets = sorted(set(_integer(b, "budget") for b in budgets))
    rhos = sorted(set(float(r) for r in rhos))
    if not budgets or not rhos or any(not np.isfinite(r) or not 0 <= r <= 1 for r in rhos):
        raise ValueError("provide budgets >= 0 and rhos in [0, 1]")
    rng = np.random.default_rng(seed)
    theory = _theory_examples(rng)
    records, analytic_errors = [], []
    for trial in range(trials):
        w = rng.normal(0, 0.5, size=(3, dimension))
        b, target = rng.normal(0, 0.2, size=3), rng.normal(0, 0.5, size=3)
        x, initial = rng.uniform(-0.8, 0.8, size=dimension), rng.normal(0, 0.3, size=dimension)
        observed = linear_gradient(w, b, x, target)
        theta = np.concatenate((w.ravel(), b))
        exact = analytic_recovery(observed, 3, dimension)
        analytic_errors.append(float(np.linalg.norm(exact - x)))
        for strategy in ("taylor", "magnitude", "random"):
            masks = nested_masks(theta, observed, rhos, strategy, seed + trial)
            for rho, mask in zip(rhos, masks):
                for budget in budgets:
                    recovered = invert_gradient(w, b, target, observed, mask, initial, budget)
                    delta = recovered["reconstruction"] - x
                    records.append({"trial": trial, "strategy": strategy, "rho": rho,
                                    "selected_coordinates": int(mask.sum()),
                                    "total_coordinates": observed.size,
                                    "effective_retention": float(mask.mean()), "budget": budget,
                                    "performed_updates": recovered["optimizer_updates"],
                                    "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(x), 1e-12)),
                                    "mse": float(np.mean(delta ** 2)),
                                    "matching_objective": recovered["matching_objective"]})
    return {
        "schema_version": 1,
        "config": {"seed": seed, "dimension": dimension, "outputs": 3, "trials": trials,
                   "rhos": rhos, "budgets": budgets, "learning_rate": 0.05},
        "scope": {"data": "seeded synthetic vectors only", "label": "known",
                  "gradient": "single-record squared-error weight and bias gradients",
                  "precision": "float64; no quantization", "budget_unit": "Adam optimizer updates",
                  "mask": "post-hoc nested supports from one fixed gradient; indices exposed",
                  "optimizer_selection_constraints": False,
                  "epiplexity_estimated": False, "mdl_estimated": False,
                  "privacy_guarantee": False},
        "theory": theory, "analytic_full_gradient_l2_errors": analytic_errors,
        "inversion": records,
    }
