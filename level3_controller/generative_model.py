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
from pathlib import Path
from typing import Mapping

import numpy as np
from pymdp import utils as pymdp_utils

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


def calibrate_pymdp_A_from_data(
    h5_path: Path,
    concentration_prior: float = 1.0,
) -> list[np.ndarray]:
    """
    Derive pymdp A matrices (likelihood P(obs|mode)) from training_data.h5.

    Theoretical role: empirical Bayes initialisation of the A matrix — the
    central scientific claim of the generative model.  Each entry
    A[obs_i, mode_j] = P(obs_i | hidden_mode_j) is estimated by counting
    how often each discrete observation category co-occurs with each
    inferred behavioral mode across the training sweep.

    Mode inference heuristic (grounded in Drosophila plume-tracking literature,
    Álvarez-Salvado et al. 2018; Demir et al. 2020):
      SURGE  — on-axis, mean odor > 0.4  (upwind runs)
      CAST   — off-axis gradient or low odor  (lateral search)
      STOP   — d_food < 0.03 and c_avg > 0.5  (feeder approach halt)
      AVOID  — not present in training sweep; kept at flat prior

    Observation discretisation uses the same thresholds as
    ActiveInferenceController._discretize_obs() so inference and calibration
    are consistent at runtime.

    Parameters
    ----------
    h5_path : Path
        Path to training_data.h5 produced by generate_training_data.py.
    concentration_prior : float
        Dirichlet pseudocount added to every (obs, mode) cell before
        normalisation.  Default 1.0 = Laplace smoothing.

    Returns
    -------
    list of four (n_obs_states, 4) float arrays — A matrices in the same
    order as build_pymdp_matrices(): [A_odor, A_gradient, A_obstacle, A_food].
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for A matrix calibration: pip install h5py"
        ) from exc

    # Dirichlet concentration counts — shape (n_obs_states, n_modes=4)
    # Obstacle modality has no training data; stays at flat prior.
    pA_odor     = np.ones((3, 4), dtype=float) * concentration_prior
    pA_gradient = np.ones((3, 4), dtype=float) * concentration_prior
    pA_obstacle = np.ones((3, 4), dtype=float) * concentration_prior  # uniform
    pA_food     = np.ones((3, 4), dtype=float) * concentration_prior

    # Discretisation thresholds — must match _discretize_obs() in active_inference.py
    _ODOR_MED   = 0.3
    _ODOR_HIGH  = 0.6
    _GRAD_THRESH = 0.10
    _FOOD_NEAR  = 0.20
    _FOOD_AT    = 0.05

    with h5py.File(h5_path, "r") as f:
        for pos_key in f:
            if pos_key == "fit":
                continue
            # Parse position from group key: pos00_x-0.050_y0.000
            parts = pos_key.split("_")
            try:
                x = float(parts[1][1:])   # strip leading 'x'
                y = float(parts[2][1:])   # strip leading 'y'
            except (IndexError, ValueError):
                continue
            d_food = float(np.hypot(x, y))  # food source at arena origin

            for cond_key in f[pos_key]:
                grp = f[pos_key][cond_key]
                if "c_left_trace" not in grp or "c_right_trace" not in grp:
                    continue

                c_left_trace  = grp["c_left_trace"][:]
                c_right_trace = grp["c_right_trace"][:]

                c_avg   = float(0.5 * (c_left_trace.mean() + c_right_trace.mean()))
                delta_c = float((c_left_trace - c_right_trace).mean())

                # Infer ground-truth mode from position + sensory context.
                # AVOID is excluded from training data (no obstacle conditions
                # in the sweep), so it retains only the flat prior mass.
                if d_food < 0.03 and c_avg > 0.5:
                    mode = 3  # STOP
                elif c_avg > 0.4:
                    mode = 0  # SURGE — strong on-axis odor, near source
                elif abs(delta_c) > 0.05 or c_avg > 0.1:
                    mode = 1  # CAST — lateral gradient or low/intermittent odor
                else:
                    mode = 0  # SURGE default for upwind far-field positions

                # Discretise each observation modality
                odor_idx = 0 if c_avg < _ODOR_MED else (1 if c_avg < _ODOR_HIGH else 2)
                grad_idx = (
                    0 if delta_c >  _GRAD_THRESH else
                    (2 if delta_c < -_GRAD_THRESH else 1)
                )
                food_idx = (
                    2 if d_food < _FOOD_AT else
                    (1 if d_food < _FOOD_NEAR else 0)
                )

                pA_odor[odor_idx, mode]     += 1.0
                pA_gradient[grad_idx, mode] += 1.0
                pA_food[food_idx, mode]     += 1.0

    def _norm_cols(pA: np.ndarray) -> np.ndarray:
        """Normalise each column to sum to 1 (valid probability distribution)."""
        col_sums = pA.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)  # guard zero columns
        return pA / col_sums

    # Convert to obj_array for pymdp 0.0.7.1 Agent compatibility
    result = pymdp_utils.obj_array(4)
    result[0] = _norm_cols(pA_odor)
    result[1] = _norm_cols(pA_gradient)
    result[2] = _norm_cols(pA_obstacle)
    result[3] = _norm_cols(pA_food)
    return result


def build_pymdp_matrices(A_calibrated: list[np.ndarray] | None = None):
    """
    Construct A, B, C, D arrays for the Level 3 pymdp discrete agent.

    Returns (A, B, C, D) ready for ``pymdp.agent.Agent(A=A, B=B, C=C, D=D)``.

    Parameters
    ----------
    A_calibrated : list of np.ndarray or None
        If provided, replaces the hand-coded A matrices with data-grounded
        values from ``calibrate_pymdp_A_from_data()``.  Must be a 4-element
        list in the order [A_odor, A_gradient, A_obstacle, A_food], each
        shaped (n_obs_states, 4).  When None the fallback hard-coded matrices
        are used (suitable only for quick tests without training data).

    Discrete state space (Level 3 only):
      Hidden state factor 0 — Behavioral mode:
        SURGE(0)  CAST(1)  AVOID(2)  STOP(3)

      Observation modalities:
        0  Odor level     LOW(0)         MED(1)         HIGH(2)
        1  Gradient dir   LEFT_HIGHER(0) BALANCED(1)    RIGHT_HIGHER(2)
        2  Obstacle prox  CLEAR(0)       NEAR(1)        CLOSE(2)
        3  Food proximity FAR(0)         NEAR(1)        AT(2)

      Actions: 4, one per behavioral mode.

    The continuous Level 2 state (10D diagonal Gaussian) is NOT represented
    here.  The discretisation of Level 2 belief means into observation indices
    is performed by ActiveInferenceController._discretize_obs().
    """
    # ------------------------------------------------------------------
    # A matrices — P(o_m | s), shape (n_obs_states_m, n_hidden_states)
    # Each column gives P(obs | mode=col); must sum to 1.0 along axis 0.
    # ------------------------------------------------------------------

    if A_calibrated is not None:
        # Data-grounded path: use empirical Bayes A from training_data.h5
        A = A_calibrated
    else:
        # Fallback hard-coded matrices — for quick tests only.
        # Replace by passing A_calibrated from calibrate_pymdp_A_from_data().
        # Modality 0: Odor level [LOW, MED, HIGH]  x  [SURGE, CAST, AVOID, STOP]
        A_odor = np.array([
            [0.10, 0.30, 0.60, 0.05],  # LOW
            [0.30, 0.50, 0.30, 0.15],  # MED
            [0.60, 0.20, 0.10, 0.80],  # HIGH
        ], dtype=float)

        # Modality 1: Gradient [LEFT_HIGHER, BALANCED, RIGHT_HIGHER]
        A_gradient = np.array([
            [0.20, 0.40, 0.40, 0.15],  # LEFT_HIGHER
            [0.60, 0.20, 0.20, 0.70],  # BALANCED
            [0.20, 0.40, 0.40, 0.15],  # RIGHT_HIGHER
        ], dtype=float)

        # Modality 2: Obstacle proximity [CLEAR, NEAR, CLOSE]
        A_obstacle = np.array([
            [0.70, 0.80, 0.10, 0.80],  # CLEAR
            [0.20, 0.15, 0.40, 0.15],  # NEAR
            [0.10, 0.05, 0.50, 0.05],  # CLOSE
        ], dtype=float)

        # Modality 3: Food proximity [FAR, NEAR, AT]
        A_food = np.array([
            [0.50, 0.70, 0.70, 0.05],  # FAR
            [0.40, 0.25, 0.25, 0.15],  # NEAR
            [0.10, 0.05, 0.05, 0.80],  # AT
        ], dtype=float)

        A = [A_odor, A_gradient, A_obstacle, A_food]

    A_obj = pymdp_utils.obj_array(len(A))
    for i, a in enumerate(A):
        A_obj[i] = a

    # ------------------------------------------------------------------
    # B matrix — P(s' | s, a), shape (n_states, n_states, n_actions)
    # B[s', s, a] = P(next_mode=s' | current_mode=s, action=a)
    # Selecting action a (= a mode) drives hidden state toward that mode
    # with probability 0.85; residual mass spreads uniformly (0.05 each).
    # ------------------------------------------------------------------
    n_states = 4
    n_actions = 4
    B_modes = np.zeros((n_states, n_states, n_actions), dtype=float)
    for a in range(n_actions):
        for s in range(n_states):
            B_modes[a, s, a] = 0.85
            for sp in range(n_states):
                if sp != a:
                    B_modes[sp, s, a] = 0.05

    B = [B_modes]

    B_obj = pymdp_utils.obj_array(1)
    B_obj[0] = B_modes

    # ------------------------------------------------------------------
    # C vectors — log prior preferences over observations
    # Higher values = more preferred outcomes.
    # ------------------------------------------------------------------
    C_odor     = np.array([-2.0,  0.0,  2.0], dtype=float)  # prefer HIGH odor
    C_gradient = np.array([ 0.0,  1.0,  0.0], dtype=float)  # prefer BALANCED
    C_obstacle = np.array([ 2.0, -1.0, -3.0], dtype=float)  # strongly prefer CLEAR
    C_food     = np.array([-1.0,  1.0,  3.0], dtype=float)  # strongly prefer AT

    C = [C_odor, C_gradient, C_obstacle, C_food]

    C_obj = pymdp_utils.obj_array(4)
    for i, c in enumerate(C):
        C_obj[i] = c

    # ------------------------------------------------------------------
    # D vector — uniform prior over initial behavioral modes
    # ------------------------------------------------------------------
    D = [np.ones(n_states, dtype=float) / n_states]

    D_obj = pymdp_utils.obj_array(1)
    D_obj[0] = np.ones(n_states, dtype=float) / n_states

    return A_obj, B_obj, C_obj, D_obj