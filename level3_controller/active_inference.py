"""
Level 3 Controller — Active Inference Belief Updater (Phase 3a)
Maintains factored beliefs at two levels:
  q(s^2) — continuous 10D Gaussian over walking-task state
  q(s^3) — discrete categorical over 4 behavioral modes

Level 2 state vector (indices):
  0  x           body forward position (m)
  1  y           body lateral position (m)
  2  theta       heading angle (rad)
  3  c_left      left-antenna odor concentration (a.u.)
  4  c_right     right-antenna odor concentration (a.u.)
  5  delta_c     bilateral gradient c_left - c_right (a.u.)
  6  w_x         wind forward component (m/s)
  7  w_y         wind lateral component (m/s)
  8  d_obs       distance to nearest obstacle (m)
  9  d_food      distance to food target (m)

Level 3 modes (in order): SURGE, CAST, AVOID, STOP

Belief update rule (per 20 ms cycle):
  1. Predict: kinematic propagation using last motor command
  2. Correct body/env: fuse MuJoCo observation for x,y,theta,d_obs,d_food
  3. Correct neural: use PP-GLM log-likelihood to update c_left,c_right,delta_c
  4. Update Level 3: re-weight modes based on mu and sigma of s^2

Biological motivation: factored continuous + discrete beliefs mirror the
suspected hierarchical structure of Drosophila navigation circuitry, where
descending neurons encode discrete locomotor modes while premotor circuits
integrate continuous sensory state. Active inference framing unifies both
levels under free energy minimisation without requiring separate reward signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class BehavioralMode(IntEnum):
    SURGE = 0
    CAST  = 1
    AVOID = 2
    STOP  = 3


# ── Index constants for the 10D state vector ────────────────────────────────
IDX_X       = 0
IDX_Y       = 1
IDX_THETA   = 2
IDX_C_LEFT  = 3
IDX_C_RIGHT = 4
IDX_DELTA_C = 5
IDX_WX      = 6
IDX_WY      = 7
IDX_D_OBS   = 8
IDX_D_FOOD  = 9
STATE_DIM   = 10


# ── Thresholds for mode-probability heuristics ──────────────────────────────
_ODOR_HIGH   = 0.5    # c_left or c_right above this → plume contact
_D_OBS_CLOSE = 0.15   # m — obstacle proximity threshold for AVOID
_D_FOOD_STOP = 0.05   # m — food distance threshold for STOP


@dataclass
class BodyEnvState:
    """
    Level 2 Gaussian belief: N(mu, diag(sigma^2)) over the 10D walking-task state.
    Uses diagonal covariance for tractability; off-diagonal terms are deferred.
    """
    mu:    np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    sigma: np.ndarray = field(default_factory=lambda: np.ones(STATE_DIM) * 0.1)

    # ── convenience properties ──
    @property
    def x(self)       -> float: return float(self.mu[IDX_X])
    @property
    def y(self)       -> float: return float(self.mu[IDX_Y])
    @property
    def theta(self)   -> float: return float(self.mu[IDX_THETA])
    @property
    def c_left(self)  -> float: return float(self.mu[IDX_C_LEFT])
    @property
    def c_right(self) -> float: return float(self.mu[IDX_C_RIGHT])
    @property
    def delta_c(self) -> float: return float(self.mu[IDX_DELTA_C])
    @property
    def w_x(self)     -> float: return float(self.mu[IDX_WX])
    @property
    def w_y(self)     -> float: return float(self.mu[IDX_WY])
    @property
    def d_obs(self)   -> float: return float(self.mu[IDX_D_OBS])
    @property
    def d_food(self)  -> float: return float(self.mu[IDX_D_FOOD])


@dataclass
class TaskState:
    """Level 3 categorical belief over 4 behavioral modes."""
    probs: np.ndarray = field(default_factory=lambda: np.ones(4) / 4.0)

    def mode(self) -> BehavioralMode:
        """Return the MAP behavioral mode."""
        return BehavioralMode(int(np.argmax(self.probs)))


class ActiveInferenceController:
    """
    Maintains q(s^2) and q(s^3) and performs predict-correct belief updates
    on a 20 ms cycle. Delegates action selection to policy.py.
    """

    # Process noise (std) on each state dimension per 20 ms step
    _PROCESS_NOISE = np.array([
        0.005,  # x     (small: walking speed ~0.01 m/step)
        0.005,  # y
        0.05,   # theta (heading can change ~0.05 rad per step)
        0.1,    # c_left   (odor is volatile)
        0.1,    # c_right
        0.1,    # delta_c
        0.02,   # w_x
        0.02,   # w_y
        0.05,   # d_obs
        0.01,   # d_food  (slow change)
    ])

    # Observation noise (std) for body/environment sensor fusion
    _OBS_NOISE_BODY = np.array([
        0.002,  # x
        0.002,  # y
        0.01,   # theta
        0.0,    # not observed by MuJoCo body obs
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,   # d_obs
        0.005,  # d_food
    ])

    # Observation noise for antennal / odor channels
    _OBS_NOISE_ODOR = np.array([0.05, 0.05, 0.05])  # c_left, c_right, delta_c

    def __init__(self):
        self.body_state = BodyEnvState()
        self.task_state = TaskState()
        self._last_motor: dict = {}  # last applied motor command (for predict step)

    # ── Belief update ────────────────────────────────────────────────────────

    def update_beliefs(
        self,
        mujoco_obs: dict,
        ppglm_log_lik: float | None = None,
        odor_obs: np.ndarray | None = None,
    ) -> None:
        """
        Full predict-correct belief update (20 ms cycle).

        Parameters
        ----------
        mujoco_obs : dict with keys 'x','y','theta','d_obs','d_food','w_x','w_y'
        ppglm_log_lik : float — PP-GLM log-likelihood of the recent spike window
            (used as a scalar correction weight on odor uncertainty)
        odor_obs : (3,) array [c_left, c_right, delta_c] from antennal sensors;
            if None, the odor dimensions are left at the predicted values
        """
        # 1. Predict: propagate uncertainty with walking kinematics
        self._predict()

        # 2. Correct with body/env observation
        self._correct_body(mujoco_obs)

        # 3. Correct with neural bridge (odor dimensions)
        if odor_obs is not None:
            self._correct_odor(odor_obs, ppglm_log_lik)

        # 4. Update Level 3 mode probabilities
        self._update_task_state()

    def _predict(self) -> None:
        """
        Propagate mu and sigma using walking kinematic model + process noise.
        The kinematic update: forward motion dx = v * cos(theta) * dt,
        dy = v * sin(theta) * dt, heading unchanged until motor command applied.
        For the predict step, we assume zero velocity (conservative prior).
        """
        # Inflate uncertainty with process noise (sigma grows each step)
        self.body_state.sigma = np.sqrt(
            self.body_state.sigma ** 2 + self._PROCESS_NOISE ** 2
        )

    def _correct_body(self, obs: dict) -> None:
        """
        Kalman-style correction for directly observed body/env dimensions.
        Observed dims: x, y, theta, d_obs, d_food, w_x, w_y.
        """
        obs_indices = {
            IDX_X:     obs.get("x", None),
            IDX_Y:     obs.get("y", None),
            IDX_THETA: obs.get("theta", None),
            IDX_D_OBS: obs.get("d_obs", None),
            IDX_D_FOOD:obs.get("d_food", None),
            IDX_WX:    obs.get("w_x", None),
            IDX_WY:    obs.get("w_y", None),
        }
        noise = self._OBS_NOISE_BODY

        for idx, val in obs_indices.items():
            if val is None or noise[idx] == 0.0:
                continue
            r = noise[idx] ** 2
            k = self.body_state.sigma[idx] ** 2 / (self.body_state.sigma[idx] ** 2 + r)
            self.body_state.mu[idx] += k * (val - self.body_state.mu[idx])
            self.body_state.sigma[idx] = np.sqrt((1 - k) * self.body_state.sigma[idx] ** 2)

    def _correct_odor(self, odor_obs: np.ndarray, log_lik: float | None) -> None:
        """
        Correct odor dimensions (c_left, c_right, delta_c) with antennal obs.
        The PP-GLM log-likelihood modulates observation precision: a higher
        log-likelihood (more informative bridge) tightens the correction.
        """
        precision_scale = 1.0
        if log_lik is not None:
            # Heuristic: more negative log-lik → less confident correction
            precision_scale = float(np.clip(1.0 + 0.1 * log_lik, 0.1, 5.0))

        odor_indices = [IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]
        for j, idx in enumerate(odor_indices):
            r = (self._OBS_NOISE_ODOR[j] / precision_scale) ** 2
            k = self.body_state.sigma[idx] ** 2 / (self.body_state.sigma[idx] ** 2 + r)
            self.body_state.mu[idx] += k * (float(odor_obs[j]) - self.body_state.mu[idx])
            self.body_state.sigma[idx] = np.sqrt((1 - k) * self.body_state.sigma[idx] ** 2)

        # Keep delta_c consistent with left/right means
        self.body_state.mu[IDX_DELTA_C] = (
            self.body_state.mu[IDX_C_LEFT] - self.body_state.mu[IDX_C_RIGHT]
        )

    def _update_task_state(self) -> None:
        """
        Re-weight Level 3 mode probabilities from current Level 2 belief mean.
        Uses a heuristic softmax over mode-specific desirability scores.
        """
        mu = self.body_state.mu
        c_avg  = 0.5 * (mu[IDX_C_LEFT] + mu[IDX_C_RIGHT])
        d_obs  = mu[IDX_D_OBS]
        d_food = mu[IDX_D_FOOD]

        # Raw scores (higher = more likely this mode is active)
        s_surge = c_avg - self.body_state.sigma[IDX_C_LEFT]   # high odor, low uncertainty
        s_cast  = self.body_state.sigma[IDX_C_LEFT] - c_avg   # high uncertainty, low odor
        s_avoid = max(0.0, _D_OBS_CLOSE - d_obs) * 10.0       # close obstacle
        s_stop  = max(0.0, _D_FOOD_STOP - d_food) * 10.0 + (c_avg - 0.3) * 2.0

        scores = np.array([s_surge, s_cast, s_avoid, s_stop])
        # Softmax to probabilities
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        self.task_state.probs = exp_scores / exp_scores.sum()
