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
  3. Correct neural: fuse a spike-derived posterior over c_left,c_right,delta_c
  4. Update Level 3: softmax mode posterior from Level 2 log potentials

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

from level2_bridge.ppglm import OdorPosterior


class BehavioralMode(IntEnum):
    SURGE = 0
    CAST = 1
    AVOID = 2
    STOP = 3


# Index constants for the 10D state vector
IDX_X = 0
IDX_Y = 1
IDX_THETA = 2
IDX_C_LEFT = 3
IDX_C_RIGHT = 4
IDX_DELTA_C = 5
IDX_WX = 6
IDX_WY = 7
IDX_D_OBS = 8
IDX_D_FOOD = 9
STATE_DIM = 10


_ODOR_HIGH = 0.5
_D_OBS_CLOSE = 0.15
_D_FOOD_STOP = 0.05


@dataclass
class BodyEnvState:
    """Level 2 Gaussian belief: N(mu, diag(sigma^2)) over the 10D task state."""

    mu: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    sigma: np.ndarray = field(default_factory=lambda: np.ones(STATE_DIM) * 0.1)

    @property
    def x(self) -> float:
        return float(self.mu[IDX_X])

    @property
    def y(self) -> float:
        return float(self.mu[IDX_Y])

    @property
    def theta(self) -> float:
        return float(self.mu[IDX_THETA])

    @property
    def c_left(self) -> float:
        return float(self.mu[IDX_C_LEFT])

    @property
    def c_right(self) -> float:
        return float(self.mu[IDX_C_RIGHT])

    @property
    def delta_c(self) -> float:
        return float(self.mu[IDX_DELTA_C])

    @property
    def w_x(self) -> float:
        return float(self.mu[IDX_WX])

    @property
    def w_y(self) -> float:
        return float(self.mu[IDX_WY])

    @property
    def d_obs(self) -> float:
        return float(self.mu[IDX_D_OBS])

    @property
    def d_food(self) -> float:
        return float(self.mu[IDX_D_FOOD])


@dataclass
class TaskState:
    """Level 3 categorical belief over behavioral modes."""

    probs: np.ndarray = field(default_factory=lambda: np.ones(4) / 4.0)

    def mode(self) -> BehavioralMode:
        return BehavioralMode(int(np.argmax(self.probs)))


class ActiveInferenceController:
    """Maintains q(s^2) and q(s^3) on a 20 ms control cycle."""

    _PROCESS_NOISE = np.array([
        0.005,
        0.005,
        0.05,
        0.1,
        0.1,
        0.1,
        0.02,
        0.02,
        0.05,
        0.01,
    ])

    _OBS_NOISE_BODY = np.array([
        0.002,
        0.002,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.005,
    ])

    def __init__(self):
        self.body_state = BodyEnvState()
        self.task_state = TaskState()
        self._last_motor: dict = {}

    def update_beliefs(
        self,
        mujoco_obs: dict,
        spike_posterior: OdorPosterior | None = None,
    ) -> None:
        """Full predict-correct belief update for one controller cycle."""
        self._predict()
        self._correct_body(mujoco_obs)
        if spike_posterior is not None:
            self._correct_odor(spike_posterior)
        self._update_task_state()

    def _predict(self) -> None:
        self.body_state.sigma = np.sqrt(
            self.body_state.sigma ** 2 + self._PROCESS_NOISE ** 2
        )

    def _correct_body(self, obs: dict) -> None:
        obs_indices = {
            IDX_X: obs.get("x", None),
            IDX_Y: obs.get("y", None),
            IDX_THETA: obs.get("theta", None),
            IDX_D_OBS: obs.get("d_obs", None),
            IDX_D_FOOD: obs.get("d_food", None),
            IDX_WX: obs.get("w_x", None),
            IDX_WY: obs.get("w_y", None),
        }

        for idx, val in obs_indices.items():
            if val is None or self._OBS_NOISE_BODY[idx] == 0.0:
                continue
            prior_var = self.body_state.sigma[idx] ** 2
            obs_var = self._OBS_NOISE_BODY[idx] ** 2
            gain = prior_var / (prior_var + obs_var)
            self.body_state.mu[idx] += gain * (float(val) - self.body_state.mu[idx])
            self.body_state.sigma[idx] = np.sqrt(max((1.0 - gain) * prior_var, 1e-6))

    def _correct_odor(self, spike_posterior: OdorPosterior) -> None:
        odor_indices = [IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]
        for j, idx in enumerate(odor_indices):
            prior_var = self.body_state.sigma[idx] ** 2
            obs_var = max(float(spike_posterior.sigma[j]) ** 2, 1e-6)
            gain = prior_var / (prior_var + obs_var)
            self.body_state.mu[idx] += gain * (
                float(spike_posterior.mean[j]) - self.body_state.mu[idx]
            )
            self.body_state.sigma[idx] = np.sqrt(max((1.0 - gain) * prior_var, 1e-6))

        self.body_state.mu[IDX_DELTA_C] = (
            self.body_state.mu[IDX_C_LEFT] - self.body_state.mu[IDX_C_RIGHT]
        )

    def _mode_log_potentials(self) -> np.ndarray:
        mu = self.body_state.mu
        sigma = self.body_state.sigma
        c_avg = 0.5 * (mu[IDX_C_LEFT] + mu[IDX_C_RIGHT])
        odor_uncertainty = float(np.sum(sigma[[IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]]))
        gradient_strength = abs(mu[IDX_DELTA_C])
        d_obs = mu[IDX_D_OBS]
        d_food = mu[IDX_D_FOOD]

        return np.array([
            2.5 * c_avg - 1.5 * d_food - 0.5 * odor_uncertainty,
            1.5 * odor_uncertainty + 0.5 * gradient_strength - c_avg,
            10.0 * max(0.0, _D_OBS_CLOSE - d_obs) - 0.5 * c_avg,
            8.0 * max(0.0, _D_FOOD_STOP - d_food) + 2.0 * c_avg - 0.5 * odor_uncertainty,
        ])

    def _update_task_state(self) -> None:
        scores = self._mode_log_potentials()
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        self.task_state.probs = exp_scores / exp_scores.sum()