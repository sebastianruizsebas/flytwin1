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

from utils.accelerator import xp, to_xp, to_np

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

# ── Pre-computed integer lag indices (eliminates Python loops at runtime) ─────
# Stimulus lags: 0-based index into u_sens_history counted from the end.
# A lag_idx of L means u_sens_history[-(L+1)].  Lags >= MAX_STIM_LAG_MS are
# out of range for a MAX_STIM_LAG_MS-length history and are masked to 0.
_STIM_LAG_IDX: np.ndarray = np.array(
    [int(round(lag)) for lag in _STIM_LAGS], dtype=np.intp
)
_STIM_LAG_VALID: np.ndarray = _STIM_LAG_IDX < MAX_STIM_LAG_MS
# Clamped indices that are safe to use even for out-of-range lags (masked away)
_STIM_LAG_SAFE: np.ndarray = np.where(_STIM_LAG_VALID, _STIM_LAG_IDX, 0)

# Spike-history lags: lag_idx=0 is excluded (can't reference "this" spike).
# lag_idx L means spike_history[-L].  Lags > MAX_HIST_LAG_MS are masked to 0.
_HIST_LAG_IDX: np.ndarray = np.array(
    [int(round(lag)) for lag in _HIST_LAGS], dtype=np.intp
)
_HIST_LAG_VALID: np.ndarray = (
    (_HIST_LAG_IDX > 0) & (_HIST_LAG_IDX <= MAX_HIST_LAG_MS)
)
_HIST_LAG_SAFE: np.ndarray = np.where(_HIST_LAG_VALID, _HIST_LAG_IDX, 1)  # >=1 to avoid index 0


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

    Fully vectorized: no Python loops.  Pre-computed index arrays
    (_STIM_LAG_SAFE, _HIST_LAG_SAFE) replace the per-lag for-loops.
    Works with plain numpy arrays; the return value is always numpy so it
    can be fed directly into scipy or Brian2 without conversion.

    Parameters
    ----------
    u_sens_history : (MAX_STIM_LAG_MS,) array, most recent last.
    spike_history  : (MAX_HIST_LAG_MS,) binary array, most recent last.
    heading        : current heading angle (rad).
    c_left, c_right: bilateral odor concentrations.
    wind_angle     : wind direction relative to body heading (rad).

    Returns
    -------
    np.ndarray shape (24,)
    """
    u = np.asarray(u_sens_history)
    s = np.asarray(spike_history)

    # Vectorized stimulus lookup: u[-(lag+1)] for each lag in one index op
    stim_raw  = u[-(1 + _STIM_LAG_SAFE)]          # (10,)
    stim_feats = np.where(_STIM_LAG_VALID, stim_raw, 0.0)

    # Vectorized spike-history lookup: s[-lag] for each lag in one index op
    hist_raw   = s[-_HIST_LAG_SAFE]               # (10,)
    hist_feats = np.where(_HIST_LAG_VALID, hist_raw, 0.0)

    return np.concatenate([
        [1.0],
        stim_feats,
        hist_feats,
        [heading, c_left - c_right, wind_angle],
    ])


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


def build_design_matrix_batch(
    c_left_arr: np.ndarray,
    c_right_arr: np.ndarray,
    spike_hist_mat: np.ndarray,
    heading_arr: np.ndarray,
    wind_angle_arr: np.ndarray,
):
    """
    Build design matrices for C candidate odor states × W time steps at once.

    This is the GPU-ready batch version used by ``ppglm.infer_odor_posterior``
    to replace its 169-iteration Python grid loop.  The computation is a set
    of pure array operations with no Python-level iteration over candidates or
    timesteps, making it JIT-compilable with JAX.

    Parameters
    ----------
    c_left_arr  : (C,) array of candidate left-antenna concentrations.
    c_right_arr : (C,) array of candidate right-antenna concentrations.
    spike_hist_mat : (W, MAX_HIST_LAG_MS) spike history at each timestep.
    heading_arr : (W,) heading angle at each timestep (rad).
    wind_angle_arr : (W,) wind angle relative to heading at each timestep (rad).

    Returns
    -------
    xp array of shape (C, W, N_COLS=24).

    Biological note
    ---------------
    For the odor-posterior grid search, u_sens is treated as a constant drive
    equal to 1.5 × (c_left + c_right) for each candidate.  This reflects the
    assumption that a candidate odor state would have driven a constant mean
    current to the HH neuron over the observation window — consistent with how
    ``_candidate_design_rows`` worked in the scalar-loop version.
    """
    c_left_arr  = to_xp(np.asarray(c_left_arr,  dtype=float))  # (C,)
    c_right_arr = to_xp(np.asarray(c_right_arr, dtype=float))  # (C,)
    s_mat = to_xp(np.asarray(spike_hist_mat, dtype=float))      # (W, H)
    h_arr = to_xp(np.asarray(heading_arr,    dtype=float))      # (W,)
    w_arr = to_xp(np.asarray(wind_angle_arr, dtype=float))      # (W,)

    import numpy as _np  # local alias to use numpy index arrays in both backends
    C = c_left_arr.shape[0]
    W = h_arr.shape[0]

    # ── Stimulus features (C, W, N_STIM_BASIS) ───────────────────────────────
    # For a constant-drive history = 1.5*(c_left+c_right), all lag lookups
    # return that drive value (for valid lags) or 0 (for out-of-range lags).
    drive = 1.5 * (c_left_arr + c_right_arr)          # (C,)
    stim_scale = to_xp(_STIM_LAG_VALID.astype(float)) # (10,) mask
    # (C, 1, 10) × (1, 1, 10) broadcast → (C, 1, 10) → expand to (C, W, 10)
    stim_3d = (drive[:, None, None] * stim_scale[None, None, :])  # (C, 1, 10)
    stim_3d = xp.broadcast_to(stim_3d, (C, W, N_STIM_BASIS))

    # ── Spike-history features (C, W, N_HIST_BASIS) ──────────────────────────
    # History is the same for all candidates; varies per timestep.
    hist_raw  = s_mat[:, -_HIST_LAG_SAFE]             # (W, 10) — numpy fancy index
    hist_mask = to_xp(_HIST_LAG_VALID.astype(float))  # (10,)
    hist_masked = to_xp(to_np(hist_raw)) * hist_mask[None, :]  # (W, 10)
    hist_3d = xp.broadcast_to(hist_masked[None, :, :], (C, W, N_HIST_BASIS))

    # ── Scalar columns ────────────────────────────────────────────────────────
    baseline_3d = xp.ones((C, W, 1))

    heading_3d = xp.broadcast_to(h_arr[None, :, None], (C, W, 1))

    delta_c = (c_left_arr - c_right_arr)[:, None, None]  # (C, 1, 1)
    delta_3d = xp.broadcast_to(delta_c, (C, W, 1))

    wind_3d = xp.broadcast_to(w_arr[None, :, None], (C, W, 1))

    # ── Concatenate → (C, W, 24) ─────────────────────────────────────────────
    # Must copy broadcast-only views before concatenation (JAX handles this
    # automatically; numpy requires it for contiguous output).
    return xp.concatenate([
        baseline_3d,   # col 0
        stim_3d,       # cols 1–10
        hist_3d,       # cols 11–20
        heading_3d,    # col 21
        delta_3d,      # col 22
        wind_3d,       # col 23
    ], axis=-1)  # (C, W, 24)
