"""
Level 2 Bridge — NPE (Neural Posterior Estimation) Trainer and Inference (Phase 2d)
Offline training: learn p(theta | beta_glm) using sbi NPE.
Online inference: single forward pass < 5 ms per call.

Training data: N = 1e5–1e6 pairs (theta, beta_summary)
  theta       — biophysical parameters, e.g. [g_KA, g_Na, g_CaL]
  beta_summary — top-K SS-weighted GLM coefficients from sum_of_slopes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Offline training
# ---------------------------------------------------------------------------

def train_npe(
    theta: np.ndarray,
    x_summary: np.ndarray,
    theta_min: np.ndarray,
    theta_max: np.ndarray,
    out_path: str = "data/npe_posterior.pkl",
) -> None:
    """
    Train NPE normalizing flow and pickle the posterior object.
    Uses sbi.inference.NPE with BoxUniform prior.
    Stub — implement in Phase 2d.
    """
    raise NotImplementedError("TODO Phase 2d: sbi NPE training loop")


# ---------------------------------------------------------------------------
# Online inference
# ---------------------------------------------------------------------------

class NPEInferenceEngine:
    """Wraps a trained sbi posterior for fast online amortized inference."""

    def __init__(self, posterior_path: str):
        self.posterior_path = posterior_path
        self._posterior = None  # loaded lazily

    def load(self) -> None:
        """Load pickled posterior. Call once at startup."""
        raise NotImplementedError("TODO Phase 2d: load posterior from disk")

    def infer(self, beta_summary: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Return (n_samples, n_params) posterior samples given observed summary stats.
        Must complete in < 5 ms on target hardware.
        """
        raise NotImplementedError("TODO Phase 2d: posterior.sample()")
