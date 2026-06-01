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

import numpy as np
from scipy.optimize import minimize


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
