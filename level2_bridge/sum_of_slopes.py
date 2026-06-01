"""
Level 2 Bridge — Sum of Slopes Feature Weighting (Phase 2c)
Ranks GLM coefficients by how much they vary across the biophysical parameter
axis. High SS => coefficient carries biophysically informative signal.

SS_k = sum_i |beta_{i+1,k} - beta_{i,k}|

Top-K coefficients (default K=10) are passed to NPE as summary statistics.
"""
from __future__ import annotations

import numpy as np


def sum_of_slopes(betas: np.ndarray) -> np.ndarray:
    """
    Compute SS for each coefficient dimension.

    Parameters
    ----------
    betas : (M, D) array — M conditions, D coefficients each

    Returns
    -------
    ss : (D,) array of importance scores
    """
    diffs = np.diff(betas, axis=0)
    return np.sum(np.abs(diffs), axis=0)


def select_top_k(betas: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (top_k_indices, summary_matrix) where summary_matrix is (M, k).
    Summary statistics ready for NPE input.
    """
    ss = sum_of_slopes(betas)
    idx = np.argsort(ss)[::-1][:k]
    return idx, betas[:, idx]
