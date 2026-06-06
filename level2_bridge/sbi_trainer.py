"""
Level 2 Bridge — NPE (Neural Posterior Estimation) Trainer and Inference (Phase 2d)

Workflow (must be run in this order):
  1. OFFLINE — run train_npe_from_data() once to build and save the posterior:
         python -c "from level2_bridge.sbi_trainer import train_npe_from_data; \\
                    train_npe_from_data()"
     This reads betas_all.npy + cond_grid.npy, selects top-K SS-weighted
     summary statistics, trains an NPE normalising flow, and pickles the
     posterior to data/npe_posterior.pkl (~2-5 min on CPU).

  2. ONLINE — at simulation startup, load the posterior once:
         engine = NPEInferenceEngine("data/npe_posterior.pkl")
         engine.load()
     Then on every 20 ms control step pass the current PP-GLM beta vector:
         theta_samples = engine.infer(beta_obs)   # < 5 ms
     theta_samples are (n_samples, 3) draws from p(g_KA, g_Na, g_CaL | beta_obs).
     Pass the posterior mean to HHNeuron to update conductances online.

Training data already present in data/spikes/:
  betas_all.npy  — (M, 24) beta matrix across conductance conditions
  cond_grid.npy  — (M, 3)  [g_KA, g_Na, g_CaL] table (theta)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .sum_of_slopes import select_top_k


# Default paths (relative to repo root)
_DEFAULT_BETAS_PATH  = Path("data/spikes/betas_all.npy")
_DEFAULT_COND_PATH   = Path("data/spikes/cond_grid.npy")
_DEFAULT_OUT_PATH    = Path("data/npe_posterior.pkl")
_TOP_K               = 10   # number of SS-selected summary statistics


# ---------------------------------------------------------------------------
# Offline training  (run once, saves posterior to disk)
# ---------------------------------------------------------------------------

def train_npe_from_data(
    betas_path: str | Path = _DEFAULT_BETAS_PATH,
    cond_grid_path: str | Path = _DEFAULT_COND_PATH,
    out_path: str | Path = _DEFAULT_OUT_PATH,
    top_k: int = _TOP_K,
    num_rounds: int = 1,
) -> None:
    """
    Train an NPE normalising flow on the betas_all / cond_grid assets produced
    by generate_training_data.py and pickle the posterior to disk.

    Parameters
    ----------
    betas_path     : path to betas_all.npy  — (M, 24) GLM beta matrix
    cond_grid_path : path to cond_grid.npy  — (M, 3) [g_KA, g_Na, g_CaL]
    out_path       : where to write the pickled sbi posterior object
    top_k          : number of SS-ranked summary statistics to use
    num_rounds     : sbi training rounds (1 = standard NPE-A/B/C)

    Theoretical role
    ----------------
    NPE learns the amortized posterior p(θ | x) where θ = [g_KA, g_Na, g_CaL]
    and x = top-K SS-weighted GLM coefficients.  Once trained, each online
    inference call is a single normalising-flow forward pass (~1 ms) rather
    than re-running the HH simulator.
    """
    try:
        import torch
        from sbi.inference import NPE
        from sbi.utils import BoxUniform
    except ImportError as exc:
        raise ImportError(
            "sbi and torch are required for NPE training: "
            "pip install sbi torch"
        ) from exc

    betas     = np.load(betas_path)        # (M, 24)
    cond_grid = np.load(cond_grid_path)    # (M, 3)

    # Select top-K informative summary statistics via sum-of-slopes
    top_k_idx, summary = select_top_k(betas, k=top_k)  # summary: (M, K)

    theta = torch.tensor(cond_grid, dtype=torch.float32)
    x     = torch.tensor(summary,   dtype=torch.float32)

    # Prior bounds derived from the conductance sweep in generate_training_data.py
    theta_min = torch.tensor(cond_grid.min(axis=0), dtype=torch.float32)
    theta_max = torch.tensor(cond_grid.max(axis=0), dtype=torch.float32)
    prior = BoxUniform(low=theta_min, high=theta_max)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"posterior": posterior, "top_k_idx": top_k_idx}, f)

    print(f"NPE posterior saved to {out_path}  (top-K indices: {top_k_idx})")


def train_npe(
    theta: np.ndarray,
    x_summary: np.ndarray,
    theta_min: np.ndarray,
    theta_max: np.ndarray,
    out_path: str = "data/npe_posterior.pkl",
) -> None:
    """
    Train NPE from arbitrary (theta, x_summary) arrays.
    Prefer train_npe_from_data() which auto-loads the project assets.
    """
    try:
        import torch
        from sbi.inference import NPE
        from sbi.utils import BoxUniform
    except ImportError as exc:
        raise ImportError("pip install sbi torch") from exc

    prior = BoxUniform(
        low=torch.tensor(theta_min, dtype=torch.float32),
        high=torch.tensor(theta_max, dtype=torch.float32),
    )
    inference = NPE(prior=prior)
    inference.append_simulations(
        torch.tensor(theta, dtype=torch.float32),
        torch.tensor(x_summary, dtype=torch.float32),
    )
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"posterior": posterior, "top_k_idx": None}, f)
    print(f"NPE posterior saved to {out}")


# ---------------------------------------------------------------------------
# Online inference  (load once at startup, call per control step)
# ---------------------------------------------------------------------------

class NPEInferenceEngine:
    """
    Wraps a trained sbi posterior for fast online amortized inference.

    Usage
    -----
    At simulation startup (once):
        engine = NPEInferenceEngine("data/npe_posterior.pkl")
        engine.load()

    Every 20 ms control step:
        theta_samples = engine.infer(beta_obs)   # (n_samples, 3)
        g_ka_mean, g_na_mean, g_cal_mean = theta_samples.mean(axis=0)
    """

    def __init__(self, posterior_path: str | Path = _DEFAULT_OUT_PATH):
        self.posterior_path = Path(posterior_path)
        self._posterior = None
        self._top_k_idx: np.ndarray | None = None

    def load(self) -> None:
        """
        Load pickled posterior from disk.  Call once at simulation startup.
        Raises FileNotFoundError if train_npe_from_data() has not been run yet.
        """
        if not self.posterior_path.exists():
            raise FileNotFoundError(
                f"No trained posterior found at {self.posterior_path}. "
                "Run: python -c \"from level2_bridge.sbi_trainer import "
                "train_npe_from_data; train_npe_from_data()\""
            )
        with open(self.posterior_path, "rb") as f:
            bundle = pickle.load(f)
        self._posterior = bundle["posterior"]
        self._top_k_idx = bundle.get("top_k_idx")
        print(f"NPE posterior loaded from {self.posterior_path}")

    def infer(self, beta_obs: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Return (n_samples, n_params) posterior samples given a (24,) beta vector.

        The top-K summary statistics are selected using the same indices chosen
        during training, so the observation space matches exactly.

        Parameters
        ----------
        beta_obs  : (24,) observed PP-GLM beta vector from the current trial
        n_samples : number of posterior samples to draw (default 100)

        Returns
        -------
        (n_samples, 3) float32 array of [g_KA, g_Na, g_CaL] samples
        """
        if self._posterior is None:
            raise RuntimeError("Call engine.load() before engine.infer()")

        import torch

        x = beta_obs
        if self._top_k_idx is not None:
            x = x[self._top_k_idx]   # select same K indices used at train time

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        samples = self._posterior.sample((n_samples,), x=x_t)
        return samples.numpy()
