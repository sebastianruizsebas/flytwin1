"""
Level 2 Bridge — Motor Readout Layer

Maps connectome motoneuron spike populations to locomotor commands in three
structured layers:

  Layer 1 — Pool layer (R matrix)
    Assign each motoneuron body ID from neurons.csv.gz to one of 8 functional
    motor pools: L/R × {propulse, retract, lift, abduct}.  Pool firing rate r_t
    is a low-pass-filtered mean spike rate (Hz, normalised to [0,1]).

  Layer 2 — Locomotor basis projection
    z_t = B @ r_t    (shape: 4,)
    z[0]  forward drive   = symmetric propulsion - retraction
    z[1]  yaw bias        = left propulsion asymmetry (positive → left turn)
    z[2]  sidestep bias   = left abduction asymmetry (positive → left)
    z[3]  stop tone       = retraction dominance (positive → tendency to stop)

  Layer 3 — Emergent mode inference
    Discrete BehavioralMode label inferred from the z_t pattern, not decoded
    from a single neuron.  Used as a diagnostic summary and as a drop-in
    replacement for the heuristic mode_to_motor_command() path.

Biological motivation:
  Drosophila motoneurons innervate specific muscle groups that produce
  stereotyped leg movements.  Protractors / extensors drive the forward swing
  phase; retractors / flexors drive stance pull-back; levators clear the leg
  during swing; lateral muscles steer.  Left/right asymmetry in homologous
  pools generates turning.  This layer preserves those structural constraints
  rather than learning a dense black-box decoder.

Integration:
  When the full connectome is imported and Level 1 is expanded to a population,
  replace mode_to_motor_command() with MotorReadout.step().  Until then,
  the existing mode-based policy remains the fallback.

References:
  Azevedo et al. 2024 — whole-fly connectome motoneuron atlas
  Demir et al. 2020 — Drosophila walking kinematics
  Ramdya et al. 2023 — leg motoneuron function
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Pool layout ──────────────────────────────────────────────────────────────

N_POOLS = 8
POOL_NAMES: List[str] = [
    "L_propulse",   # 0  left-side protraction + tibial extension
    "R_propulse",   # 1  right-side protraction + tibial extension
    "L_retract",    # 2  left-side retraction + tibial flexion
    "R_retract",    # 3  right-side retraction + tibial flexion
    "L_lift",       # 4  left-side levation (swing phase clearance)
    "R_lift",       # 5  right-side levation
    "L_abduct",     # 6  left-side lateral abduction
    "R_abduct",     # 7  right-side lateral abduction
]

# Which pool indices belong to each side (used for left/right diagnostics)
_LEFT_POOLS  = [0, 2, 4, 6]
_RIGHT_POOLS = [1, 3, 5, 7]


# ── Default regex patterns for pool assignment ──────────────────────────────
# Applied to the 'type' and 'instance' columns of neurons.csv.gz.
# Designed to work across hemibrain (v1.2.1) and FlyWire female conventions.
# Override with dataset-specific patterns where labelling differs.

DEFAULT_POOL_PATTERNS: Dict[str, List[str]] = {
    "L_propulse": [
        r"(?i)MN.*[Ll][123]?.*protract",
        r"(?i)MN.*[Ll][123]?.*ext",
        r"(?i)leg.*[Ll].*protract",
        r"(?i)T[123].*[Ll].*protract",
    ],
    "R_propulse": [
        r"(?i)MN.*[Rr][123]?.*protract",
        r"(?i)MN.*[Rr][123]?.*ext",
        r"(?i)leg.*[Rr].*protract",
        r"(?i)T[123].*[Rr].*protract",
    ],
    "L_retract": [
        r"(?i)MN.*[Ll][123]?.*retract",
        r"(?i)MN.*[Ll][123]?.*flex",
        r"(?i)leg.*[Ll].*retract",
        r"(?i)T[123].*[Ll].*retract",
    ],
    "R_retract": [
        r"(?i)MN.*[Rr][123]?.*retract",
        r"(?i)MN.*[Rr][123]?.*flex",
        r"(?i)leg.*[Rr].*retract",
        r"(?i)T[123].*[Rr].*retract",
    ],
    "L_lift": [
        r"(?i)MN.*[Ll][123]?.*levat",
        r"(?i)leg.*[Ll].*levat",
        r"(?i)T[123].*[Ll].*levat",
    ],
    "R_lift": [
        r"(?i)MN.*[Rr][123]?.*levat",
        r"(?i)leg.*[Rr].*levat",
        r"(?i)T[123].*[Rr].*levat",
    ],
    "L_abduct": [
        r"(?i)MN.*[Ll][123]?.*abduct",
        r"(?i)MN.*[Ll][123]?.*lateral",
        r"(?i)leg.*[Ll].*abduct",
    ],
    "R_abduct": [
        r"(?i)MN.*[Rr][123]?.*abduct",
        r"(?i)MN.*[Rr][123]?.*lateral",
        r"(?i)leg.*[Rr].*abduct",
    ],
}


# ── Locomotor basis matrix B (4 × N_POOLS) ──────────────────────────────────
#
# z_t = B @ r_t
#
# Structural constraints encoded directly:
#   - Symmetric propulsion drives forward movement
#   - Left/right propulsion asymmetry drives yaw
#   - Retraction opposes forward and contributes to stop tone
#   - Lateral abduction asymmetry drives sidestep
#   - Levation does not appear in the command basis (it is internal to swing)
#
#           L_pr  R_pr  L_ret R_ret L_lft R_lft L_ab  R_ab
_B_DEFAULT = np.array([
    [0.5,   0.5,  -0.3, -0.3,  0.0,  0.0,  0.0,  0.0],  # forward
    [0.7,  -0.7,  -0.5,  0.5,  0.0,  0.0,  0.0,  0.0],  # yaw (+ = left)
    [0.0,   0.0,   0.0,  0.0,  0.0,  0.0,  1.0, -1.0],  # sidestep (+ = left)
    [-0.5, -0.5,   0.5,  0.5,  0.0,  0.0,  0.0,  0.0],  # stop tone
], dtype=float)


# ── Thresholds for emergent mode inference ──────────────────────────────────
_STOP_THRESHOLD      = 0.40  # z[3] above this → STOP
_SIDESTEP_THRESHOLD  = 0.30  # |z[2]| above this → AVOID candidate
_CAST_YAW_THRESHOLD  = 0.25  # |z[1]| above this with weak forward → CAST
_SURGE_FORWARD_MIN   = 0.25  # z[0] must exceed this for SURGE

# Firing rate above which a pool is treated as saturated (Hz)
_RATE_SATURATION_HZ = 200.0


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MotorPool:
    """Named functional motor pool with assigned neuron body IDs."""

    name: str
    body_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def size(self) -> int:
        return int(len(self.body_ids))


@dataclass(frozen=True)
class MotorState:
    """Snapshot of motor readout output for one time step."""

    pool_rates: np.ndarray    # (N_POOLS,) normalised firing rates
    z: np.ndarray             # (4,) locomotor basis
    command: dict             # flybody-compatible command dict
    mode: object              # emergent BehavioralMode


# ── Main readout class ───────────────────────────────────────────────────────

class MotorReadout:
    """
    Converts motoneuron spike activity to locomotor commands.

    Parameters
    ----------
    pools : list of MotorPool
        Ordered list of 8 functional motor pools.  Missing pools are filled
        with empty stubs.
    body_id_index : np.ndarray, shape (N,)
        Global body ID ordering — must match the column ordering of spike arrays
        passed to step().
    B : np.ndarray, shape (4, N_POOLS), optional
        Locomotor basis matrix.  Defaults to the anatomically structured basis.
    tau_ms : float
        Low-pass filter time constant (ms) for smoothing pool rates.
    """

    def __init__(
        self,
        pools: List[MotorPool],
        body_id_index: np.ndarray,
        B: Optional[np.ndarray] = None,
        tau_ms: float = 20.0,
        learning_rate: float = 1e-3,
    ):
        self.pools = _pad_pools(pools)
        self.body_id_index = np.asarray(body_id_index, dtype=np.int64)
        self.B = B.copy() if B is not None else _B_DEFAULT.copy()
        self.tau_ms = float(tau_ms)
        self.learning_rate = float(learning_rate)

        # Build per-pool index arrays into the global spike vector once
        id_to_pos: Dict[int, int] = {
            int(bid): i for i, bid in enumerate(self.body_id_index)
        }
        self._pool_indices: List[np.ndarray] = []
        for pool in self.pools:
            positions = np.array(
                [id_to_pos[int(bid)] for bid in pool.body_ids if int(bid) in id_to_pos],
                dtype=int,
            )
            self._pool_indices.append(positions)

        # Low-pass filter state
        self._r_smoothed = np.zeros(N_POOLS)
        # Last z_t cached for B adaptation
        self._last_z: Optional[np.ndarray] = None

    # ── Public interface ─────────────────────────────────────────────────────

    def step(
        self,
        spike_window: np.ndarray,
        dt_ms: float,
    ) -> MotorState:
        """
        Process one spike window and return the locomotor command.

        Parameters
        ----------
        spike_window : (W, N) binary array — W time bins × N neurons in the
            order of body_id_index.  A (N,) 1-D array is also accepted and
            treated as a single-bin window.
        dt_ms : float — duration of one time bin in ms.

        Returns
        -------
        MotorState with pool_rates, z, command, and emergent mode.
        """
        r_raw = self._raw_pool_rates(spike_window, dt_ms)
        alpha = dt_ms / (self.tau_ms + dt_ms)
        self._r_smoothed = (1.0 - alpha) * self._r_smoothed + alpha * r_raw

        z = self.B @ self._r_smoothed
        self._last_z = z.copy()
        command = _z_to_command(z)
        mode = _infer_mode(z)

        return MotorState(
            pool_rates=self._r_smoothed.copy(),
            z=z.copy(),
            command=command,
            mode=mode,
        )

    def adapt_from_error(self, position_error: np.ndarray) -> None:
        """
        Online adaptation of the B matrix from local positional error.

        Biological rationale: spinal / VNC circuits refine the mapping from
        motoneuron pool activity to locomotor output through error-driven
        Hebbian plasticity.  Here we implement a simple gradient step:

            ΔB = -η * error_outer(position_error, r_smoothed)

        where position_error = [Δforward, Δyaw, Δsidestep, Δstop] is the
        signed discrepancy between the intended and observed movement
        (computed externally from body-state deltas), and r_smoothed is the
        current pool firing rate vector.

        The outer product maps each pool's contribution to each command axis
        according to the observed error, tightening columns of B that
        mispredicted the outcome.  The anatomical sign constraints encoded in
        _B_DEFAULT are softly preserved by clipping.

        Parameters
        ----------
        position_error : (4,) array — [forward_err, yaw_err, sidestep_err,
            stop_err].  Positive = body moved less than commanded.
        """
        if self._last_z is None or self.learning_rate == 0.0:
            return
        error = np.asarray(position_error, dtype=float)
        # Gradient of MSE loss: dL/dB = error ⊗ r (outer product)
        dB = np.outer(error, self._r_smoothed)
        self.B -= self.learning_rate * dB
        # Clip to [-2, 2] to prevent runaway weights
        np.clip(self.B, -2.0, 2.0, out=self.B)

    def raw_pool_rates(
        self, spike_window: np.ndarray, dt_ms: float
    ) -> np.ndarray:
        """Return un-smoothed per-pool firing rates (Hz), normalised to [0,1]."""
        return self._raw_pool_rates(spike_window, dt_ms)

    def pool_summary(self) -> dict:
        """Return pool names and sizes for inspection."""
        return {p.name: p.size for p in self.pools}

    def reset(self) -> None:
        """Reset the low-pass filter state and cached z."""
        self._r_smoothed[:] = 0.0
        self._last_z = None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _raw_pool_rates(
        self, spike_window: np.ndarray, dt_ms: float
    ) -> np.ndarray:
        """Compute mean firing rate per pool, normalised by saturation rate."""
        if spike_window.ndim == 1:
            spike_counts = spike_window.astype(float)
            n_bins = 1
        else:
            spike_counts = spike_window.sum(axis=0).astype(float)
            n_bins = spike_window.shape[0]

        window_s = float(n_bins) * dt_ms * 1e-3

        rates = np.zeros(N_POOLS)
        for g, indices in enumerate(self._pool_indices):
            n_cells = len(indices)
            if n_cells == 0 or window_s <= 0.0:
                continue
            pool_count = float(spike_counts[indices].sum())
            rate_hz = pool_count / (n_cells * window_s)
            rates[g] = min(rate_hz, _RATE_SATURATION_HZ) / _RATE_SATURATION_HZ

        return rates


# ── Factory functions ────────────────────────────────────────────────────────

def assign_motor_pools(
    neuron_df: pd.DataFrame,
    pool_patterns: Optional[Dict[str, List[str]]] = None,
) -> List[MotorPool]:
    """
    Assign connectome neurons to motor pools using regex patterns.

    Parameters
    ----------
    neuron_df : DataFrame from neurons.csv.gz.
        Must contain 'bodyId' and at least one of 'type' or 'instance'.
    pool_patterns : optional dict mapping pool name → list of regex strings.
        Defaults to DEFAULT_POOL_PATTERNS.

    Returns
    -------
    list of MotorPool in POOL_NAMES order.

    Notes
    -----
    Each neuron is assigned to at most one pool: the first pool whose pattern
    matches wins.  Neurons that match no pattern are not assigned to any pool
    and remain in the full connectome scaffold without contributing to the
    motor readout.
    """
    if pool_patterns is None:
        pool_patterns = DEFAULT_POOL_PATTERNS

    type_col = neuron_df.get(
        "type", pd.Series("", index=neuron_df.index)
    ).fillna("")
    inst_col = neuron_df.get(
        "instance", pd.Series("", index=neuron_df.index)
    ).fillna("")
    combined = type_col + "|" + inst_col

    # Track already-assigned body IDs to enforce single-pool assignment
    assigned: set = set()
    pools: List[MotorPool] = []

    for name in POOL_NAMES:
        patterns = pool_patterns.get(name, [])
        if not patterns:
            pools.append(MotorPool(name=name))
            continue
        full_pattern = "|".join(f"(?:{p})" for p in patterns)
        mask = combined.str.contains(full_pattern, regex=True, na=False)
        candidate_ids = neuron_df.loc[mask, "bodyId"].to_numpy(dtype=np.int64)
        new_ids = np.array(
            [bid for bid in candidate_ids if int(bid) not in assigned],
            dtype=np.int64,
        )
        assigned.update(int(bid) for bid in new_ids)
        pools.append(MotorPool(name=name, body_ids=new_ids))

    return pools


def load_motor_readout(
    connectome_dir: str | Path,
    pool_patterns: Optional[Dict[str, List[str]]] = None,
    B: Optional[np.ndarray] = None,
    tau_ms: float = 20.0,
) -> MotorReadout:
    """
    Build a MotorReadout from saved connectome assets.

    Parameters
    ----------
    connectome_dir : path to data/connectome/ (containing neurons.csv.gz
        and body_ids.npy produced by import_connectome.py).
    pool_patterns : optional override for pool assignment regex patterns.
    B : optional override for the locomotor basis matrix.
    tau_ms : low-pass filter time constant (ms).

    Returns
    -------
    MotorReadout ready for use in the closed loop.
    """
    root = Path(connectome_dir)
    neuron_df = pd.read_csv(root / "neurons.csv.gz")
    body_id_index = np.load(root / "body_ids.npy")
    pools = assign_motor_pools(neuron_df, pool_patterns)
    return MotorReadout(
        pools=pools,
        body_id_index=body_id_index,
        B=B,
        tau_ms=tau_ms,
    )


# ── Internal utilities ───────────────────────────────────────────────────────

def _pad_pools(pools: List[MotorPool]) -> List[MotorPool]:
    """Ensure exactly N_POOLS pools, adding empty stubs for missing entries."""
    name_to_pool = {p.name: p for p in pools}
    return [name_to_pool.get(name, MotorPool(name=name)) for name in POOL_NAMES]


def _z_to_command(z: np.ndarray) -> dict:
    """
    Convert 4D locomotor basis vector to a flybody-compatible command dict.

    The yaw component is scaled to rad/s assuming a maximum yaw rate of 5 rad/s.
    """
    forward   = float(z[0])
    yaw       = float(z[1])
    sidestep  = float(z[2])
    stop_tone = float(z[3])

    is_active = stop_tone < _STOP_THRESHOLD

    return {
        "forward_speed": float(np.clip(forward, 0.0, 1.0)),
        "yaw_rate":      float(np.clip(yaw * 5.0, -5.0, 5.0)),   # → rad/s
        "sidestep":      float(np.clip(sidestep, -1.0, 1.0)),
        "active":        bool(is_active),
    }


def _infer_mode(z: np.ndarray) -> object:
    """
    Infer a discrete BehavioralMode from the 4D locomotor basis.

    Import is deferred to avoid a circular dependency with level3_controller.
    Precedence: STOP > AVOID > CAST > SURGE.
    """
    from level3_controller.active_inference import BehavioralMode  # deferred

    forward, yaw, sidestep, stop_tone = (
        float(z[0]), float(z[1]), float(z[2]), float(z[3])
    )

    if stop_tone > _STOP_THRESHOLD:
        return BehavioralMode.STOP

    if abs(sidestep) > _SIDESTEP_THRESHOLD:
        return BehavioralMode.AVOID

    if abs(yaw) > _CAST_YAW_THRESHOLD and forward < _SURGE_FORWARD_MIN:
        return BehavioralMode.CAST

    return BehavioralMode.SURGE
