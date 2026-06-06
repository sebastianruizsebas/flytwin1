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

from pymdp.agent import Agent

from level2_bridge.ppglm import OdorPosterior
from .generative_model import (
    ObservationModel,
    PreferredOutcomeModel,
    StateBelief,
    TransitionModel,
    build_pymdp_matrices,
    calibrate_pymdp_A_from_data,
)


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

# Thresholds for discretising the Level 2 continuous belief into the
# categorical observation indices expected by the Level 3 pymdp agent.
_OBS_ODOR_MED_THRESH  = 0.3   # c_avg below → LOW, above → MED or HIGH
_OBS_ODOR_HIGH_THRESH = 0.6   # c_avg above → HIGH
_OBS_GRADIENT_THRESH  = 0.10  # |delta_c| below → BALANCED
_OBS_OBS_NEAR_THRESH  = 0.30  # d_obs below → NEAR
_OBS_OBS_CLOSE_THRESH = 0.15  # d_obs below → CLOSE
_OBS_FOOD_NEAR_THRESH = 0.20  # d_food below → NEAR
_OBS_FOOD_AT_THRESH   = 0.05  # d_food below → AT


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

    _CTRL_DT_S = 0.02

    # The initial filtering prior p(s_0) should reflect what is already known
    # before new evidence arrives: start pose and mean wind are usually known
    # from arena setup, while plume contact is intermittent and therefore kept
    # as a low-concentration, high-uncertainty prior.
    _INITIAL_SIGMA = np.array([
        0.02,
        0.02,
        0.05,
        0.25,
        0.25,
        0.30,
        0.05,
        0.05,
        0.15,
        0.10,
    ])
    _LOW_ODOR_PRIOR = 0.05
    _DEFAULT_WIND_PRIOR = np.array([0.2, 0.0])
    _DEFAULT_OBS_PRIOR = 1.0

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

    # Direct body/environment observation model. Odor concentrations are kept
    # out of this body term for now so the PP-GLM remains the explicit odor
    # likelihood bridge and we avoid double-counting plume evidence.
    _BODY_OBS_KEYS = ("x", "y", "theta", "w_x", "w_y", "d_obs", "d_food")
    _BODY_OBS_INDICES = (IDX_X, IDX_Y, IDX_THETA, IDX_WX, IDX_WY, IDX_D_OBS, IDX_D_FOOD)
    _BODY_OBS_SIGMA = np.array([
        0.002,
        0.002,
        0.01,
        0.02,
        0.02,
        0.02,
        0.005,
    ])

    def __init__(self, initial_obs: dict | None = None,
                 h5_path: "Path | None" = None):
        self.initial_state_prior = self._build_initial_state_prior(initial_obs)
        self.predictive_prior = self.initial_state_prior.copy()
        self.body_state = BodyEnvState(
            mu=self.initial_state_prior.mu.copy(),
            sigma=self.initial_state_prior.sigma.copy(),
        )
        self.task_state = TaskState()
        # Stores the last applied motor command as an efference-copy signal used
        # by P(s'|s,a). This lets the controller predict how its own locomotor
        # output should move the body before new sensory evidence arrives.
        self._last_motor: dict = {
            "forward_speed": 0.0,
            "yaw_rate": 0.0,
            "sidestep": 0.0,
            "active": True,
        }
        self.preferred_outcomes = self._build_preferred_outcomes()
        self.transition_model = TransitionModel(process_noise=self._PROCESS_NOISE)
        self.observation_model = ObservationModel(
            body_obs_keys=self._BODY_OBS_KEYS,
            body_obs_indices=self._BODY_OBS_INDICES,
            body_obs_sigma=self._BODY_OBS_SIGMA,
            odor_indices=(IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C),
        )
        self.last_body_log_likelihood = 0.0
        self.last_neural_log_likelihood = 0.0
        self.last_total_log_likelihood = 0.0
        self.pymdp_agent: Agent = self._build_pymdp_agent(h5_path=h5_path)
        # Cached index of the action recommended by the pymdp policy posterior.
        # Initialised to SURGE (0) as the default exploratory prior.
        self._pymdp_action_idx: int = int(BehavioralMode.SURGE)

    def set_last_motor(self, motor_command: dict | None) -> None:
        """
        Store the last applied action for the next predictive prior update.

        Theoretical underpinning: the predictive prior should condition on the
        previous action. In a biological framing this is a coarse efference-copy
        channel from the selected locomotor command back into state prediction.
        """
        cmd = motor_command or {}
        self._last_motor = {
            "forward_speed": float(cmd.get("forward_speed", 0.0)),
            "yaw_rate": float(cmd.get("yaw_rate", 0.0)),
            "sidestep": float(cmd.get("sidestep", 0.0)),
            "active": bool(cmd.get("active", True)),
        }

    @classmethod
    def _build_initial_state_prior(cls, initial_obs: dict | None) -> StateBelief:
        """
        Construct the initial filtering prior p(s_0) from known setup variables.

        Theoretical underpinning: p(s_0) should capture what the controller
        already believes before the first online evidence update.  Pose and wind
        start relatively narrow because the arena setup makes them known, while
        odor remains broad because turbulent plume encounters are intermittent.
        """
        obs = initial_obs or {}
        mu = np.zeros(STATE_DIM, dtype=float)
        sigma = cls._INITIAL_SIGMA.astype(float).copy()

        mu[IDX_X] = float(obs.get("x", 0.0))
        mu[IDX_Y] = float(obs.get("y", 0.0))
        mu[IDX_THETA] = float(obs.get("theta", 0.0))

        c_left = max(0.0, float(obs.get("c_left", cls._LOW_ODOR_PRIOR)))
        c_right = max(0.0, float(obs.get("c_right", cls._LOW_ODOR_PRIOR)))
        mu[IDX_C_LEFT] = c_left
        mu[IDX_C_RIGHT] = c_right
        mu[IDX_DELTA_C] = c_left - c_right

        mu[IDX_WX] = float(obs.get("w_x", cls._DEFAULT_WIND_PRIOR[0]))
        mu[IDX_WY] = float(obs.get("w_y", cls._DEFAULT_WIND_PRIOR[1]))
        mu[IDX_D_OBS] = float(obs.get("d_obs", cls._DEFAULT_OBS_PRIOR))

        default_d_food = float(np.hypot(mu[IDX_X], mu[IDX_Y]))
        mu[IDX_D_FOOD] = float(obs.get("d_food", default_d_food))

        return StateBelief(mu=mu, sigma=sigma)

    @staticmethod
    def _build_preferred_outcomes() -> PreferredOutcomeModel:
        """
        Build prior preferences p*(o) used by the pragmatic term in policy.py.

        Biological rationale: these are not beliefs about the current state.
        They encode the ethological objective of the walking task: keep odor on,
        stay clear of obstacles, and approach then stop at the feeder.
        """
        preferred_mu = np.zeros(STATE_DIM, dtype=float)
        preferred_mu[IDX_C_LEFT] = 0.8
        preferred_mu[IDX_C_RIGHT] = 0.8
        preferred_mu[IDX_D_OBS] = 0.3
        preferred_mu[IDX_D_FOOD] = 0.0

        preferred_weight = np.array([
            0.0,
            0.0,
            0.0,
            1.0,   # c_left
            1.0,   # c_right
            0.5,   # delta_c
            0.0,
            0.0,
            0.0,   # d_obs: zeroed until MuJoCo contact sensors are live;
                   # frozen at 1.0 otherwise biases EFE toward AVOID permanently
            3.0,   # d_food
        ])
        return PreferredOutcomeModel(
            preferred_mu=preferred_mu,
            preferred_weight=preferred_weight,
        )

    def update_beliefs(
        self,
        mujoco_obs: dict,
        spike_posterior: OdorPosterior | None = None,
    ) -> None:
        """Full predict-correct belief update for one controller cycle."""
        self._predict()
        self.last_body_log_likelihood = 0.0
        self.last_neural_log_likelihood = 0.0
        self._correct_body(mujoco_obs)
        if spike_posterior is not None:
            self._correct_odor(spike_posterior)
        self.last_total_log_likelihood = (
            self.last_body_log_likelihood + self.last_neural_log_likelihood
        )
        self._update_task_state()

    def _predict(self) -> None:
        predicted = self.transition_model.predict(
            belief=StateBelief(
                mu=self.body_state.mu.copy(),
                sigma=self.body_state.sigma.copy(),
            ),
            action=self._last_motor,
            dt=self._CTRL_DT_S,
        )
        self.predictive_prior = predicted.copy()
        self.body_state.mu = predicted.mu.copy()
        self.body_state.sigma = predicted.sigma.copy()

    def _correct_body(self, obs: dict) -> None:
        self.last_body_log_likelihood = self.observation_model.body_log_likelihood(
            obs=obs,
            state_mu=self.body_state.mu,
        )
        mu, sigma = self.observation_model.correct_body(
            state_mu=self.body_state.mu,
            state_sigma=self.body_state.sigma,
            obs=obs,
        )
        self.body_state.mu = mu
        self.body_state.sigma = sigma

    def _correct_odor(self, spike_posterior: OdorPosterior) -> None:
        self.last_neural_log_likelihood = self.observation_model.odor_log_likelihood(
            spike_posterior=spike_posterior,
            state_mu=self.body_state.mu,
        )
        mu, sigma = self.observation_model.correct_odor(
            state_mu=self.body_state.mu,
            state_sigma=self.body_state.sigma,
            spike_posterior=spike_posterior,
        )
        self.body_state.mu = mu
        self.body_state.sigma = sigma

    @classmethod
    def _build_pymdp_agent(cls, h5_path: "Path | None" = None) -> Agent:
        """
        Instantiate the Level 3 pymdp discrete active inference agent.

        When h5_path points to a completed training_data.h5 the A matrices
        are calibrated via empirical Bayes (calibrate_pymdp_A_from_data);
        otherwise the hard-coded fallback matrices are used.

        Theoretical role: A encodes P(obs|mode) — the generative model's
        claim about which sensory observations predict each behavioral mode.
        Grounding A in data makes the agent's predictions falsifiable.
        """
        A_cal = None
        if h5_path is not None:
            from pathlib import Path as _Path
            _p = _Path(h5_path)
            if _p.exists():
                try:
                    A_cal = calibrate_pymdp_A_from_data(_p)
                    import logging as _logging
                    _logging.getLogger(__name__).info(
                        "pymdp A matrices calibrated from %s", _p
                    )
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"A matrix calibration failed ({exc}); "
                        "falling back to hard-coded matrices."
                    )
        A, B, C, D = build_pymdp_matrices(A_calibrated=A_cal)
        return Agent(A=A, B=B, C=C, D=D, policy_len=1)

    @staticmethod
    def _discretize_obs(body_state: "BodyEnvState") -> list:
        """
        Map Level 2 continuous belief means to discrete observation indices
        for the Level 3 pymdp agent.

        Returns a 4-element list [odor_idx, gradient_idx, obstacle_idx, food_idx].
          odor_idx     : 0=LOW  1=MED  2=HIGH
          gradient_idx : 0=LEFT_HIGHER  1=BALANCED  2=RIGHT_HIGHER
          obstacle_idx : 0=CLEAR  1=NEAR  2=CLOSE
          food_idx     : 0=FAR  1=NEAR  2=AT
        """
        c_avg = 0.5 * (body_state.c_left + body_state.c_right)
        odor_idx = (
            0 if c_avg < _OBS_ODOR_MED_THRESH else
            (1 if c_avg < _OBS_ODOR_HIGH_THRESH else 2)
        )

        delta_c = body_state.delta_c
        if delta_c > _OBS_GRADIENT_THRESH:
            gradient_idx = 0  # left antenna higher
        elif delta_c < -_OBS_GRADIENT_THRESH:
            gradient_idx = 2  # right antenna higher
        else:
            gradient_idx = 1  # balanced

        d_obs = body_state.d_obs
        obstacle_idx = (
            0 if d_obs > _OBS_OBS_NEAR_THRESH else
            (1 if d_obs > _OBS_OBS_CLOSE_THRESH else 2)
        )

        d_food = body_state.d_food
        food_idx = (
            2 if d_food < _OBS_FOOD_AT_THRESH else
            (1 if d_food < _OBS_FOOD_NEAR_THRESH else 0)
        )

        return [odor_idx, gradient_idx, obstacle_idx, food_idx]

    def _update_task_state(self) -> None:
        """
        Update Level 3 discrete belief using pymdp variational message passing.

        1. Discretise the current Level 2 continuous belief means into the four
           categorical observation indices expected by the pymdp agent.
        2. Run one step of variational state inference (infer_states).
        3. Run policy inference / EFE computation (infer_policies).
        4. Cache the argmax action for use by policy.select_action().
        5. Mirror the pymdp hidden-state posterior into TaskState.probs so that
           existing logging / diagnostic code continues to work unchanged.
        """
        obs = self._discretize_obs(self.body_state)
        self.pymdp_agent.infer_states(obs)
        self.pymdp_agent.infer_policies()
        self.task_state.probs = np.asarray(
            self.pymdp_agent.qs[0], dtype=float
        ).copy()
        self._pymdp_action_idx = int(np.argmax(self.pymdp_agent.q_pi))