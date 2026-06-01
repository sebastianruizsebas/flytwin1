"""
Level 2 Bridge — PP-GLM with Trend-Filter Penalty (Phase 2b/2c)
Fits logistic GLM coefficients beta (24-dim) across M biophysical conditions
using a joint L1 trend-filter penalty on adjacent coefficient vectors.

Loss:
    L(beta_1,...,beta_M) = -sum_i log P(data_i | beta_i)
                         + lambda * sum_i ||beta_{i+1} - beta_i||_1

Fitting: L-BFGS-B (scipy) on flattened beta vector with soft-L1 approximation
for gradient availability (true L1 uses proximal steps, but L-BFGS-B on
smoothed L1 is accurate enough for the 24-dim case).

Online interface (Phase 2c):
  evaluate_online(spike_window, beta, u_sens_window, ...) -> log-likelihood update
  used by the closed loop every 10 ms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from utils.accelerator import xp, jit, to_np, to_xp, HAS_JAX
from .design_matrix import (
    MAX_STIM_LAG_MS, N_COLS,
    build_design_row,
    build_design_matrix_batch,
)


@dataclass(frozen=True)
class OdorPosterior:
    """Posterior summary over odor-related Level 2 state inferred from spikes."""

    mean: np.ndarray
    sigma: np.ndarray
    log_evidence: float
    map_log_likelihood: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Bernoulli log-likelihood for a single condition."""
    p = sigmoid(X @ beta)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _gaussian_log_prior(value: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-3)
    z = (value - mean) / sigma
    return float(-0.5 * np.sum(z ** 2 + np.log(2.0 * np.pi * sigma ** 2)))


def _gaussian_log_prior_batch(values, mean: np.ndarray, sigma: np.ndarray):
    """
    Vectorized Gaussian log-prior for C candidate states at once.

    Parameters
    ----------
    values : (C, 3) array of candidate [c_left, c_right, delta_c] states.
    mean   : (3,) prior mean.
    sigma  : (3,) prior std.

    Returns
    -------
    (C,) array of log-prior values.
    """
    sigma_xp = xp.maximum(to_xp(sigma), 1e-3)  # (3,)
    mean_xp  = to_xp(mean)                       # (3,)
    z = (values - mean_xp[None, :]) / sigma_xp[None, :]  # (C, 3)
    return -0.5 * xp.sum(
        z ** 2 + xp.log(2.0 * np.pi * sigma_xp[None, :] ** 2),
        axis=1,
    )  # (C,)


def _logsumexp(values) -> float:
    max_value = float(xp.max(values))
    return max_value + float(xp.log(xp.sum(xp.exp(values - max_value))))


# _candidate_design_rows is replaced by build_design_matrix_batch from
# design_matrix.py (vectorised over all grid candidates simultaneously).


def infer_odor_posterior(
    spike_window: np.ndarray,
    beta: np.ndarray,
    spike_history_window: np.ndarray,
    heading_window: np.ndarray,
    wind_angle_window: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    grid_size: int = 13,
) -> OdorPosterior:
    """
    Infer a posterior over [c_left, c_right, delta_c] from the recent spike window.

    GPU-accelerated: the 169-point (13×13) grid search is now a single batched
    matrix multiply (C×W×24) @ (24,) → (C×W), followed by vectorised log-
    likelihood and prior scoring.  No Python loop over grid candidates.
    """
    if spike_window.size == 0:
        return OdorPosterior(
            mean=prior_mean.astype(float).copy(),
            sigma=np.maximum(prior_sigma.astype(float), 1e-3),
            log_evidence=0.0,
            map_log_likelihood=0.0,
        )

    prior_mean  = prior_mean.astype(float)
    prior_sigma = np.maximum(prior_sigma.astype(float), 1e-3)

    c_mean_prior = max(0.0, 0.5 * (prior_mean[0] + prior_mean[1]))
    c_mean_sigma = max(0.1, 0.5 * (prior_sigma[0] + prior_sigma[1]))
    delta_prior  = float(prior_mean[2])
    delta_sigma  = max(0.1, float(prior_sigma[2]))

    c_max      = max(1.5, c_mean_prior + 3.0 * c_mean_sigma)
    delta_span = max(0.6, abs(delta_prior) + 3.0 * delta_sigma)

    # Build flat candidate arrays (C = grid_size^2)
    c_mean_grid = np.linspace(0.0, c_max, grid_size)
    delta_grid  = np.linspace(-delta_span, delta_span, grid_size)
    cm_mesh, dc_mesh = np.meshgrid(c_mean_grid, delta_grid, indexing='ij')  # (G, G)
    c_left_all  = np.maximum(0.0, cm_mesh.ravel() + 0.5 * dc_mesh.ravel())  # (C,)
    c_right_all = np.maximum(0.0, cm_mesh.ravel() - 0.5 * dc_mesh.ravel())  # (C,)
    states_np   = np.stack([
        c_left_all,
        c_right_all,
        c_left_all - c_right_all,
    ], axis=1)  # (C, 3)

    # ── Batched design matrix: (C, W, 24) on xp device ────────────────────────
    X_batch = build_design_matrix_batch(
        c_left_arr=c_left_all,
        c_right_arr=c_right_all,
        spike_hist_mat=spike_history_window,
        heading_arr=heading_window,
        wind_angle_arr=wind_angle_window,
    )  # (C, W, 24) on xp

    beta_xp = to_xp(np.asarray(beta, dtype=float))  # (24,)
    y_xp    = to_xp(spike_window.astype(float))      # (W,)

    # ── Batch log-likelihood: (C, W, 24) × (24,) → logits (C, W) ────────
    logits = X_batch @ beta_xp  # (C, W)
    # Numerically stable sigmoid
    p = xp.where(
        logits >= 0,
        1.0 / (1.0 + xp.exp(-logits)),
        xp.exp(logits) / (1.0 + xp.exp(logits)),
    )
    p = xp.clip(p, 1e-9, 1.0 - 1e-9)  # (C, W)

    log_lik_batch = xp.sum(
        y_xp[None, :] * xp.log(p) + (1.0 - y_xp[None, :]) * xp.log(1.0 - p),
        axis=1,
    )  # (C,)

    # ── Batch Gaussian prior ──────────────────────────────────────────────
    states_xp = to_xp(states_np)  # (C, 3)
    log_prior_batch = _gaussian_log_prior_batch(states_xp, prior_mean, prior_sigma)  # (C,)

    log_post_batch = log_lik_batch + log_prior_batch  # (C,)

    # ── Normalise and compute posterior mean / std ─────────────────────────
    log_norm   = _logsumexp(log_post_batch)
    weights_xp = xp.exp(log_post_batch - log_norm)  # (C,)

    # Bring back to numpy for output (avoids JAX device arrays escaping the module)
    weights   = to_np(weights_xp).astype(float)   # (C,)
    log_liks  = to_np(log_lik_batch).astype(float) # (C,)

    post_mean = np.sum(states_np * weights[:, None], axis=0)          # (3,)
    var       = np.sum(weights[:, None] * (states_np - post_mean) ** 2, axis=0)
    post_sigma = np.sqrt(np.maximum(var, 1e-4))

    return OdorPosterior(
        mean=post_mean,
        sigma=post_sigma,
        log_evidence=float(log_norm),
        map_log_likelihood=float(np.max(log_liks)),
    )


def _smooth_l1(z: np.ndarray, eps: float = 1e-4) -> float:
    """Huber-style smooth approximation to ||z||_1 for gradient-based optimisation."""
    return float(np.sum(np.where(np.abs(z) <= eps, z**2 / (2 * eps), np.abs(z) - eps / 2)))


def _smooth_l1_grad(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.where(np.abs(z) <= eps, z / eps, np.sign(z))


def fit_joint(
    X_list: list[np.ndarray],
    y_list: list[np.ndarray],
    lam: float = 1.0,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Jointly fit M GLMs with trend-filter penalty using L-BFGS-B.

    When JAX is available the log-likelihood gradient is computed via JAX
    autodiff (JIT-compiled, GPU-accelerated for large T).  The L-BFGS-B
    optimiser itself still runs in scipy on the host; only the inner
    gradient evaluation is accelerated.

    Parameters
    ----------
    X_list : list of (T_i, 24) arrays — design matrices per condition
    y_list : list of (T_i,) binary arrays — spike labels per condition
    lam    : L1 trend-filter penalty strength
    max_iter : max L-BFGS-B iterations

    Returns
    -------
    beta : np.ndarray shape (M, 24)
    """
    M = len(X_list)
    D = X_list[0].shape[1]  # 24

    if HAS_JAX:
        # ── JAX-accelerated gradient path ──────────────────────────────
        import jax
        import jax.numpy as jnp

        # Convert once to JAX arrays (device transfer happens here)
        Xs_jax = [jnp.asarray(X, dtype=jnp.float32) for X in X_list]
        ys_jax = [jnp.asarray(y, dtype=jnp.float32) for y in y_list]

        @jax.jit
        def _neg_ll_single(beta_i, X, y):
            p = jax.nn.sigmoid(X @ beta_i)
            p = jnp.clip(p, 1e-9, 1.0 - 1e-9)
            return -jnp.sum(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))

        _neg_ll_and_grad = jax.jit(jax.value_and_grad(_neg_ll_single))

        def objective_and_grad(flat_beta: np.ndarray):
            betas = flat_beta.reshape(M, D)
            loss = 0.0
            grad = np.zeros_like(betas)

            for i, (X, y) in enumerate(zip(Xs_jax, ys_jax)):
                beta_i = jnp.asarray(betas[i], dtype=jnp.float32)
                nll, g = _neg_ll_and_grad(beta_i, X, y)
                loss += float(nll)
                grad[i] += np.asarray(g, dtype=float)

            if M > 1:
                for i in range(M - 1):
                    diff = betas[i + 1] - betas[i]
                    pen = _smooth_l1(diff)
                    g = _smooth_l1_grad(diff)
                    loss += lam * pen
                    grad[i]     -= lam * g
                    grad[i + 1] += lam * g

            return loss, grad.ravel()

    else:
        # ── Pure-numpy gradient path (fallback) ──────────────────────────
        def objective_and_grad(flat_beta: np.ndarray):
            betas = flat_beta.reshape(M, D)
            loss = 0.0
            grad = np.zeros_like(betas)

            for i, (X, y) in enumerate(zip(X_list, y_list)):
                eta = X @ betas[i]
                p = sigmoid(eta)
                p = np.clip(p, 1e-9, 1 - 1e-9)
                loss -= float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
                grad[i] -= X.T @ (y - p)

            if M > 1:
                for i in range(M - 1):
                    diff = betas[i + 1] - betas[i]
                    pen = _smooth_l1(diff)
                    g = _smooth_l1_grad(diff)
                    loss += lam * pen
                    grad[i]     -= lam * g
                    grad[i + 1] += lam * g

            return loss, grad.ravel()

    beta0 = np.zeros(M * D)
    result = minimize(
        objective_and_grad,
        beta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5},
    )
    return result.x.reshape(M, D)


def cross_validate_lambda(
    X_list: list[np.ndarray],
    y_list: list[np.ndarray],
    lambdas: np.ndarray,
    n_folds: int = 5,
) -> float:
    """
    Return the lambda value with best held-out Bernoulli log-likelihood.

    Splits each condition's data into n_folds folds, fits on the training
    portion, evaluates on the held-out portion, and averages across folds
    and conditions.
    """
    best_lam = float(lambdas[0])
    best_score = -np.inf

    for lam in lambdas:
        fold_scores = []
        # Build per-condition fold indices
        fold_indices = [
            np.array_split(np.arange(len(y)), n_folds) for y in y_list
        ]
        for fold in range(n_folds):
            X_train, X_val, y_train, y_val = [], [], [], []
            for i, (X, y) in enumerate(zip(X_list, y_list)):
                val_idx = fold_indices[i][fold]
                train_idx = np.concatenate([fold_indices[i][f] for f in range(n_folds) if f != fold])
                X_train.append(X[train_idx])
                y_train.append(y[train_idx])
                X_val.append(X[val_idx])
                y_val.append(y[val_idx])
            betas = fit_joint(X_train, y_train, lam=lam, max_iter=200)
            score = sum(log_likelihood(X_val[i], y_val[i], betas[i]) for i in range(len(X_list)))
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_lam = float(lam)

    return best_lam


def evaluate_online(
    spike_window: np.ndarray,
    beta: np.ndarray,
    x_rows: np.ndarray,
) -> float:
    """
    Online PP-GLM likelihood evaluation for the 10 ms bridge cycle.

    Parameters
    ----------
    spike_window : (W,) binary array of recent spikes (W time bins)
    beta : (24,) coefficient vector for the current conductance condition
    x_rows : (W, 24) design matrix rows for the same window

    Returns
    -------
    log_lik : float — Bernoulli log-likelihood of the spike window under the GLM.
        Used by the Level 2 belief updater to correct odor-related dimensions.
    """
    return log_likelihood(x_rows, spike_window.astype(float), beta)
