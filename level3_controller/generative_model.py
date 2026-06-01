"""
Level 3 Controller — Explicit generative-model components.

This module now contains the three pieces needed by the controller-side
generative model developed so far:

- a diagonal-Gaussian filtering belief over hidden state
- a preferred-outcome model used by pragmatic policy scoring
- an explicit observation model P(o|s)
- an explicit transition model P(s'|s,a)

The odor-related neural evidence is still summarized by OdorPosterior for the
first pass, which keeps the PP-GLM as the Level 1 -> Level 2 bridge while
making both likelihood and transition structure explicit and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from level2_bridge.ppglm import (
    OdorPosterior,
    neural_log_likelihood as ppglm_neural_log_likelihood,
)


@dataclass
class StateBelief:
    """
    Diagonal-Gaussian filtering belief over the current hidden state.

    Theoretical role: this object represents p(s_t) or q(s_t), depending on
    where it is used in the update cycle.  It is intentionally separate from
    any preferred-outcome object so the controller does not confuse beliefs
    about the current world with task-level goals.
    """

    mu: np.ndarray
    sigma: np.ndarray

    def copy(self) -> "StateBelief":
        return StateBelief(
            mu=np.asarray(self.mu, dtype=float).copy(),
            sigma=np.asarray(self.sigma, dtype=float).copy(),
        )


@dataclass(frozen=True)
class PreferredOutcomeModel:
    """
    Prior preferences over desirable outcomes used in pragmatic policy scoring.

    Theoretical role: these parameters correspond to prior preferences over
    future outcomes, not the filtering prior over the present hidden state.
    Biological rationale: they encode the agent's ethological task demands —
    maintain odor contact, avoid collisions, and reduce distance to the feeder.
    """

    preferred_mu: np.ndarray
    preferred_weight: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "preferred_mu", np.asarray(self.preferred_mu, dtype=float))
        object.__setattr__(
            self,
            "preferred_weight",
            np.asarray(self.preferred_weight, dtype=float),
        )


@dataclass
class TransitionModel:
    """
    Action-conditioned transition model P(s'|s,a).

    Theoretical underpinning:
    active inference requires a predictive model that propagates the current
    belief through the consequences of the last action before new evidence is
    assimilated. Here we use a diagonal-Gaussian approximation with a
    deterministic mean transition and additive process noise.

    Biological rationale:
    the deterministic part acts like a coarse locomotor efference copy: if the
    fly recently surged forward or turned, the controller should already expect
    position and feeder distance to change before the next sensory correction.
    Odor and obstacle estimates are kept on a short-horizon persistence prior
    until new sensory evidence arrives, which is a good first approximation for
    an intermittently sampled turbulent plume and sparse obstacle encounters.
    """

    process_noise: np.ndarray
    walk_scale: float = 0.01
    sidestep_scale: float = 0.5
    food_position_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    reference_dt: float = 0.02
    min_sigma: float = 1e-3

    def __post_init__(self) -> None:
        self.process_noise = np.asarray(self.process_noise, dtype=float)
        self.food_position_xy = np.asarray(self.food_position_xy, dtype=float)
        if self.process_noise.ndim != 1:
            raise ValueError("process_noise must be a 1D array")
        if self.food_position_xy.shape != (2,):
            raise ValueError("food_position_xy must have shape (2,)")

    @staticmethod
    def _sanitize_action(action: Mapping[str, float] | None) -> dict[str, float | bool]:
        cmd = action or {}
        return {
            "forward_speed": float(cmd.get("forward_speed", 0.0)),
            "yaw_rate": float(cmd.get("yaw_rate", 0.0)),
            "sidestep": float(cmd.get("sidestep", 0.0)),
            "active": bool(cmd.get("active", True)),
        }

    def predict_mean(
        self,
        state_mu: np.ndarray,
        action: Mapping[str, float] | None,
        dt: float,
    ) -> np.ndarray:
        """
        Deterministic part of the transition mean.

        State layout is the project-wide 10D walking-task state:
        [x, y, theta, c_left, c_right, delta_c, w_x, w_y, d_obs, d_food].
        """
        mu = np.asarray(state_mu, dtype=float).copy()
        cmd = self._sanitize_action(action)

        if cmd["active"]:
            theta = float(mu[2])
            dx = float(cmd["forward_speed"]) * self.walk_scale * np.cos(theta) * dt
            dy = float(cmd["forward_speed"]) * self.walk_scale * np.sin(theta) * dt
            dy += float(cmd["sidestep"]) * self.walk_scale * self.sidestep_scale * dt
            dtheta = float(cmd["yaw_rate"]) * dt

            mu[0] += dx
            mu[1] += dy
            mu[2] += dtheta

        # Food distance is geometry-derived from the predicted pose. This is the
        # cleanest first-pass way to encode the task objective inside P(s'|s,a)
        # without yet needing a full flybody geometry query in the controller.
        mu[9] = float(np.hypot(mu[0] - self.food_position_xy[0], mu[1] - self.food_position_xy[1]))

        # Odor and wind states follow a short-horizon persistence prior here;
        # the observation update remains responsible for pulling them back to
        # the sensed plume and wind values on every correction cycle.
        mu[5] = mu[3] - mu[4]

        # Obstacle distance also follows persistence until explicit geometry is
        # exposed to the controller-side transition model.
        return mu

    def predict_sigma(self, state_sigma: np.ndarray, dt: float) -> np.ndarray:
        sigma = np.asarray(state_sigma, dtype=float)
        scaled_process_noise = self.process_noise * (dt / self.reference_dt)
        return np.sqrt(np.maximum(sigma, self.min_sigma) ** 2 + scaled_process_noise ** 2)

    def predict(
        self,
        belief: StateBelief,
        action: Mapping[str, float] | None,
        dt: float,
    ) -> StateBelief:
        return StateBelief(
            mu=self.predict_mean(belief.mu, action, dt),
            sigma=self.predict_sigma(belief.sigma, dt),
        )


@dataclass
class ObservationModel:
    """Explicit observation model P(o|s) for body and neural evidence."""

    body_obs_keys: tuple[str, ...]
    body_obs_indices: tuple[int, ...]
    body_obs_sigma: np.ndarray
    odor_indices: tuple[int, int, int]
    min_sigma: float = 1e-3

    def __post_init__(self) -> None:
        if len(self.body_obs_keys) != len(self.body_obs_indices):
            raise ValueError("body observation keys and indices must have the same length")
        self.body_obs_sigma = np.asarray(self.body_obs_sigma, dtype=float)
        if self.body_obs_sigma.shape != (len(self.body_obs_keys),):
            raise ValueError("body_obs_sigma must match the number of body observation keys")
        if np.any(self.body_obs_sigma <= 0.0):
            raise ValueError("body_obs_sigma values must be strictly positive")

    def expected_body_observation(self, state_mu: np.ndarray) -> dict[str, float]:
        """Return the expected body observation h(s) for the observed state dims."""
        state_mu = np.asarray(state_mu, dtype=float)
        return {
            key: float(state_mu[idx])
            for key, idx in zip(self.body_obs_keys, self.body_obs_indices)
        }

    def body_log_likelihood(self, obs: Mapping[str, float], state_mu: np.ndarray) -> float:
        """Diagonal-Gaussian log-likelihood for body/environment observations."""
        state_mu = np.asarray(state_mu, dtype=float)
        residuals = []
        variances = []
        for key, idx, sigma in zip(
            self.body_obs_keys,
            self.body_obs_indices,
            self.body_obs_sigma,
        ):
            value = obs.get(key, None)
            if value is None:
                continue
            residuals.append(float(value) - state_mu[idx])
            variances.append(max(float(sigma), self.min_sigma) ** 2)

        if not residuals:
            return 0.0

        residual_vec = np.asarray(residuals, dtype=float)
        var_vec = np.asarray(variances, dtype=float)
        return float(
            -0.5 * np.sum(residual_vec ** 2 / var_vec + np.log(2.0 * np.pi * var_vec))
        )

    def correct_body(
        self,
        state_mu: np.ndarray,
        state_sigma: np.ndarray,
        obs: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse body observations into a diagonal-Gaussian state belief."""
        mu = np.asarray(state_mu, dtype=float).copy()
        sigma = np.asarray(state_sigma, dtype=float).copy()

        for key, idx, obs_sigma in zip(
            self.body_obs_keys,
            self.body_obs_indices,
            self.body_obs_sigma,
        ):
            value = obs.get(key, None)
            if value is None:
                continue
            prior_var = max(float(sigma[idx]), self.min_sigma) ** 2
            obs_var = max(float(obs_sigma), self.min_sigma) ** 2
            gain = prior_var / (prior_var + obs_var)
            mu[idx] += gain * (float(value) - mu[idx])
            sigma[idx] = np.sqrt(max((1.0 - gain) * prior_var, self.min_sigma ** 2))

        return mu, sigma

    def odor_log_likelihood(self, spike_posterior: OdorPosterior, state_mu: np.ndarray) -> float:
        """
        Local Gaussian log-likelihood of the current odor state under the
        PP-GLM-derived posterior summary.

        This keeps the Step 1 controller API compatible with the current runner,
        which already passes an OdorPosterior object every 10 ms.
        """
        state_mu = np.asarray(state_mu, dtype=float)
        odor_state = state_mu[list(self.odor_indices)]
        post_mean = np.asarray(spike_posterior.mean, dtype=float)
        post_sigma = np.maximum(np.asarray(spike_posterior.sigma, dtype=float), self.min_sigma)
        var = post_sigma ** 2
        residual = odor_state - post_mean
        return float(-0.5 * np.sum(residual ** 2 / var + np.log(2.0 * np.pi * var)))

    def correct_odor(
        self,
        state_mu: np.ndarray,
        state_sigma: np.ndarray,
        spike_posterior: OdorPosterior,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse the PP-GLM odor posterior into the odor-related state dims."""
        mu = np.asarray(state_mu, dtype=float).copy()
        sigma = np.asarray(state_sigma, dtype=float).copy()

        for local_idx, state_idx in enumerate(self.odor_indices):
            prior_var = max(float(sigma[state_idx]), self.min_sigma) ** 2
            obs_var = max(float(spike_posterior.sigma[local_idx]), self.min_sigma) ** 2
            gain = prior_var / (prior_var + obs_var)
            mu[state_idx] += gain * (float(spike_posterior.mean[local_idx]) - mu[state_idx])
            sigma[state_idx] = np.sqrt(max((1.0 - gain) * prior_var, self.min_sigma ** 2))

        # Keep delta_c consistent with the bilateral concentrations.
        mu[self.odor_indices[2]] = mu[self.odor_indices[0]] - mu[self.odor_indices[1]]
        return mu, sigma

    def neural_log_likelihood(
        self,
        spike_window: np.ndarray,
        beta: np.ndarray,
        spike_history_window: np.ndarray,
        heading_window: np.ndarray,
        wind_angle_window: np.ndarray,
        candidate_odor_state: np.ndarray,
    ) -> float:
        """Delegate raw spike likelihood evaluation to the PP-GLM bridge."""
        return ppglm_neural_log_likelihood(
            spike_window=spike_window,
            beta=beta,
            spike_history_window=spike_history_window,
            heading_window=heading_window,
            wind_angle_window=wind_angle_window,
            candidate_odor_state=candidate_odor_state,
        )