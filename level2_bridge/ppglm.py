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

from .design_matrix import MAX_STIM_LAG_MS, N_COLS, build_design_row


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


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _candidate_design_rows(
    spike_history_window: np.ndarray,
    heading_window: np.ndarray,
    wind_angle_window: np.ndarray,
    c_left: float,
    c_right: float,
) -> np.ndarray:
    drive = 1.5 * (c_left + c_right)
    u_hist = np.full(MAX_STIM_LAG_MS, drive, dtype=float)
    rows = np.empty((len(heading_window), N_COLS), dtype=float)

    for t in range(len(heading_window)):
        rows[t] = build_design_row(
            u_sens_history=u_hist,
            spike_history=spike_history_window[t],
            heading=float(heading_window[t]),
            c_left=c_left,
            c_right=c_right,
            wind_angle=float(wind_angle_window[t]),
        )
    return rows


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

    The bridge uses a grid approximation over a low-dimensional odor latent state.
    Candidate odor causes are converted into PP-GLM design rows, scored by the
    Bernoulli spike likelihood, then combined with the Level 2 Gaussian prior.
    """
    if spike_window.size == 0:
        mean = prior_mean.astype(float).copy()
        sigma = np.maximum(prior_sigma.astype(float), 1e-3)
        return OdorPosterior(
            mean=mean,
            sigma=sigma,
            log_evidence=0.0,
            map_log_likelihood=0.0,
        )

    prior_mean = prior_mean.astype(float)
    prior_sigma = np.maximum(prior_sigma.astype(float), 1e-3)

    c_mean_prior = max(0.0, 0.5 * (prior_mean[0] + prior_mean[1]))
    c_mean_sigma = max(0.1, 0.5 * (prior_sigma[0] + prior_sigma[1]))
    delta_prior = float(prior_mean[2])
    delta_sigma = max(0.1, float(prior_sigma[2]))

    c_max = max(1.5, c_mean_prior + 3.0 * c_mean_sigma)
    delta_span = max(0.6, abs(delta_prior) + 3.0 * delta_sigma)

    c_mean_grid = np.linspace(0.0, c_max, grid_size)
    delta_grid = np.linspace(-delta_span, delta_span, grid_size)

    posterior_states = []
    log_posts = []
    map_log_likelihood = -np.inf

    spike_window = spike_window.astype(float)
    for c_mean in c_mean_grid:
        for delta_c in delta_grid:
            c_left = max(0.0, c_mean + 0.5 * delta_c)
            c_right = max(0.0, c_mean - 0.5 * delta_c)
            state = np.array([c_left, c_right, c_left - c_right], dtype=float)
            X = _candidate_design_rows(
                spike_history_window=spike_history_window,
                heading_window=heading_window,
                wind_angle_window=wind_angle_window,
                c_left=c_left,
                c_right=c_right,
            )
            candidate_log_lik = log_likelihood(X, spike_window, beta)
            candidate_log_post = candidate_log_lik + _gaussian_log_prior(state, prior_mean, prior_sigma)

            posterior_states.append(state)
            log_posts.append(candidate_log_post)
            map_log_likelihood = max(map_log_likelihood, candidate_log_lik)

    states = np.vstack(posterior_states)
    log_posts_arr = np.array(log_posts, dtype=float)
    log_norm = _logsumexp(log_posts_arr)
    weights = np.exp(log_posts_arr - log_norm)

    mean = np.sum(states * weights[:, None], axis=0)
    var = np.sum(weights[:, None] * (states - mean) ** 2, axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-4))

    return OdorPosterior(
        mean=mean,
        sigma=sigma,
        log_evidence=log_norm,
        map_log_likelihood=float(map_log_likelihood),
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

    def objective_and_grad(flat_beta: np.ndarray):
        betas = flat_beta.reshape(M, D)
        loss = 0.0
        grad = np.zeros_like(betas)

        # Negative log-likelihood sum over conditions
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            eta = X @ betas[i]
            p = sigmoid(eta)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            loss -= float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            grad[i] -= X.T @ (y - p)

        # Trend-filter penalty on adjacent beta vectors
        if M > 1:
            for i in range(M - 1):
                diff = betas[i + 1] - betas[i]
                pen = _smooth_l1(diff)
                g = _smooth_l1_grad(diff)
                loss += lam * pen
                grad[i]     -= lam * g
                grad[i + 1] += lam * g

        return loss, grad.ravel()

    # Warm-start: independent logistic fits per condition
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
