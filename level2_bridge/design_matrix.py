"""
Level 2 Bridge — Design Matrix Construction (Phase 2a)
Converts raw spike history and sensory inputs into a fixed 24-column feature
vector x_t for PP-GLM fitting.

Column layout (0-indexed):
  0        — baseline constant (intercept)
  1–10     — stimulus filter: 10 bell basis functions on u_sens scalar,
              non-linearly spaced lags 0–200 ms (denser near lag 0)
  11–20    — spike-history filter: 10 bell basis functions, lags 1–100 ms
  21       — heading theta_t (rad, from Level 2 body state)
  22       — bilateral odor gradient: delta_c = c_left - c_right
  23       — wind angle relative to heading (rad)

Biological motivation:
  Stimulus filter — captures odor-driven depolarisation timing in ORN/PN cascade.
  Spike-history filter — captures refractoriness and burst adaptation.
  Heading and delta_c — couple neural activity to the locomotor context needed
    for the EFE controller to link spikes to plume direction.
  Wind angle — encodes upwind/downwind cues that modulate SURGE vs CAST selection.
"""
from __future__ import annotations

import numpy as np

N_COLS = 24
N_STIM_BASIS = 10
N_HIST_BASIS = 10
MAX_STIM_LAG_MS = 200
MAX_HIST_LAG_MS = 100


def _bell_basis_lags(n: int, max_lag: float, nonlinear: bool = False) -> np.ndarray:
    """Return n lag centres, optionally non-linearly spaced (denser near 0)."""
    if nonlinear:
        return np.geomspace(1, max_lag, n)
    return np.linspace(0, max_lag, n)


def _bell_basis(value: float, centers: np.ndarray, width_scale: float = 1.0) -> np.ndarray:
    """
    Evaluate n bell (Gaussian) basis functions at a scalar value.
    Width of each basis is proportional to half the spacing between adjacent centres.
    """
    spacings = np.diff(centers)
    # Width for each basis: half of average adjacent spacing
    widths = np.empty(len(centers))
    widths[1:-1] = 0.5 * (spacings[:-1] + spacings[1:]) / 2.0
    widths[0] = spacings[0] / 2.0 if len(spacings) > 0 else 1.0
    widths[-1] = spacings[-1] / 2.0 if len(spacings) > 0 else 1.0
    widths = np.clip(widths * width_scale, 1e-3, None)
    return np.exp(-0.5 * ((value - centers) / widths) ** 2)


# Pre-compute lag centres (reused across all calls)
_STIM_LAGS = _bell_basis_lags(N_STIM_BASIS, MAX_STIM_LAG_MS, nonlinear=True)
_HIST_LAGS = _bell_basis_lags(N_HIST_BASIS, MAX_HIST_LAG_MS, nonlinear=False)


def build_design_row(
    u_sens_history: np.ndarray,
    spike_history: np.ndarray,
    heading: float,
    c_left: float,
    c_right: float,
    wind_angle: float,
) -> np.ndarray:
    """
    Build a single 24-element design vector x_t.

    Parameters
    ----------
    u_sens_history : np.ndarray
        (MAX_STIM_LAG_MS,) array of past sensory drive values, most recent last.
        Values outside the trial are assumed 0.
    spike_history : np.ndarray
        (MAX_HIST_LAG_MS,) binary array of past spikes, most recent last.
    heading : float
        Current heading angle (rad).
    c_left, c_right : float
        Bilateral odor concentrations after sigmoid gain.
    wind_angle : float
        Wind direction relative to body heading (rad).

    Returns
    -------
    np.ndarray shape (24,)
    """
    row = np.empty(N_COLS)
    row[0] = 1.0  # baseline constant

    # Stimulus filter (columns 1–10): project u_sens history onto bell bases
    # Each basis value is the inner product of the history at the basis lag
    stim_feats = np.zeros(N_STIM_BASIS)
    for j, lag in enumerate(_STIM_LAGS):
        lag_idx = int(round(lag))
        if lag_idx < len(u_sens_history):
            # Index from end: lag=0 is most recent
            stim_feats[j] = u_sens_history[-(lag_idx + 1)] if lag_idx < len(u_sens_history) else 0.0
    row[1:11] = stim_feats

    # Spike-history filter (columns 11–20)
    hist_feats = np.zeros(N_HIST_BASIS)
    for j, lag in enumerate(_HIST_LAGS):
        lag_idx = int(round(lag))
        if lag_idx > 0 and lag_idx <= len(spike_history):
            hist_feats[j] = spike_history[-lag_idx]
    row[11:21] = hist_feats

    row[21] = heading
    row[22] = c_left - c_right          # bilateral odor gradient delta_c
    row[23] = wind_angle                # wind relative to heading
    return row


def build_design_matrix(
    u_sens_trace: np.ndarray,
    spike_trace: np.ndarray,
    heading_trace: np.ndarray,
    c_left_trace: np.ndarray,
    c_right_trace: np.ndarray,
    wind_angle_trace: np.ndarray,
) -> np.ndarray:
    """
    Return (T, 24) design matrix for a full trial.

    All trace arrays must have length T (one value per ms time step).
    History padding is handled by zero-padding before the start of the trial.
    """
    T = len(u_sens_trace)
    X = np.zeros((T, N_COLS))

    # Pad histories with zeros before trial start
    u_padded = np.concatenate([np.zeros(MAX_STIM_LAG_MS), u_sens_trace])
    s_padded = np.concatenate([np.zeros(MAX_HIST_LAG_MS, dtype=spike_trace.dtype), spike_trace])

    for t in range(T):
        u_hist = u_padded[t: t + MAX_STIM_LAG_MS]
        s_hist = s_padded[t: t + MAX_HIST_LAG_MS]
        X[t] = build_design_row(
            u_sens_history=u_hist,
            spike_history=s_hist,
            heading=float(heading_trace[t]),
            c_left=float(c_left_trace[t]),
            c_right=float(c_right_trace[t]),
            wind_angle=float(wind_angle_trace[t]),
        )
    return X
