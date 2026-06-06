"""
Level 3 Controller — EFE Policy Selection (Phase 3b/3c)
Selects the walking mode π* that minimises Expected Free Energy every 20 ms.

G(π) = KL[q(s²|π) ‖ p(s²)]            ← pragmatic: approach food, avoid obstacles
     - E_q[log p(o_neural | s², π)]    ← epistemic: reduce odor uncertainty

Behavioral modes: SURGE | CAST | AVOID | STOP
Walking primitives replace the previous flight-oriented vocabulary.

Motor command dict keys:
  forward_speed  — normalised [0, 1]
  yaw_rate       — rad/s, positive = turn left
  sidestep       — lateral speed, normalised [−1, 1]
  active         — bool, False triggers STOP behaviour in flybody

Biological motivation: the four modes mirror identified Drosophila walking
behaviors during plume tracking (Álvarez-Salvado et al. 2018; Demir et al. 2020):
SURGE (~straight runs), CAST (~casting sweeps), STOP (~feeder approach halt).
AVOID maps onto obstacle-avoidance turning observed in free-walking flies.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from .generative_model import StateBelief
from .active_inference import (
    ActiveInferenceController,
    BehavioralMode,
    IDX_C_LEFT,
    IDX_C_RIGHT,
    IDX_D_FOOD,
    IDX_D_OBS,
    IDX_DELTA_C,
    IDX_THETA,
    _D_OBS_CLOSE,
    _D_FOOD_STOP,
    _ODOR_HIGH,
)


def _pragmatic_cost(
    mu_predicted: np.ndarray,
    sigma_predicted: np.ndarray,
    mode: BehavioralMode,
    controller: ActiveInferenceController,
) -> float:
    """
    Weighted squared deviation of predicted state from preferred outcomes.

    Diagnostic / analysis utility — no longer called in the main action-
    selection path (pymdp handles EFE internally).  Retained for logging
    and offline analysis of individual EFE terms.
    """
    pref = controller.preferred_outcomes.preferred_mu.copy()
    pref_weight = controller.preferred_outcomes.preferred_weight

    if mode == BehavioralMode.SURGE:
        # SURGE: move toward food, tolerate asymmetric odor
        pass
    elif mode == BehavioralMode.CAST:
        # CAST: reduce uncertainty — prefer a state with larger delta_c
        pref[IDX_DELTA_C] = 0.3  # some asymmetry expected during casting
    elif mode == BehavioralMode.AVOID:
        # AVOID: strongly prefer being away from obstacles
        pref[IDX_D_OBS] = pref[IDX_D_OBS] * 2.0
    elif mode == BehavioralMode.STOP:
        # STOP: food distance near zero, odor high
        pref[IDX_D_FOOD] = 0.0

    sq_dev = pref_weight * (mu_predicted - pref) ** 2
    return float(np.sum(sq_dev))


def _epistemic_value(sigma: np.ndarray, mode: BehavioralMode) -> float:
    """
    Epistemic value: prefer actions that reduce uncertainty.
    Approximated as negative total uncertainty on odor-related dimensions.
    CAST gets a bonus because lateral search reduces plume-direction uncertainty.
    """
    odor_uncertainty = float(np.sum(sigma[[IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]]))
    bonus = 0.5 if mode == BehavioralMode.CAST else 0.0
    return -(odor_uncertainty - bonus)  # negative: lower uncertainty is better


def _predict_state(
    controller: ActiveInferenceController,
    mode: BehavioralMode,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step prediction of mu and sigma under a given mode's typical action.

    Theoretical underpinning: policy evaluation should roll forward the same
    transition model used by the filtering prior, otherwise expected free energy
    is scored under a different dynamics model than the one used for inference.
    Returns (mu_pred, sigma_pred).
    """
    predicted = controller.transition_model.predict(
        belief=StateBelief(
            mu=controller.body_state.mu.copy(),
            sigma=controller.body_state.sigma.copy(),
        ),
        action=mode_to_motor_command(mode, controller),
        dt=dt,
    )
    return predicted.mu, predicted.sigma


def expected_free_energy(
    mode: BehavioralMode,
    controller: ActiveInferenceController,
) -> float:
    """
    Compute G(π) for a single behavioral mode under current beliefs.

    G = pragmatic_cost(predicted_state) - epistemic_value(predicted_uncertainty)

    Lower G → more preferred mode.
    """
    mu_pred, sigma_pred = _predict_state(controller, mode)
    pragmatic = _pragmatic_cost(mu_pred, sigma_pred, mode, controller)
    epistemic = _epistemic_value(sigma_pred, mode)
    return pragmatic + epistemic  # epistemic already negative when uncertainty low


def select_action(controller: ActiveInferenceController) -> BehavioralMode:
    """
    Return the behavioral mode with minimum continuous Expected Free Energy.

    Uses the continuous EFE (pragmatic + epistemic value evaluated under the
    current Level 2 Gaussian belief) as the policy selection criterion.
    pymdp is retained for state inference (qs update) but its policy posterior
    q_pi is not used here: the calibrated A matrix maps STOP→HIGH_odor and
    STOP→AT_food, causing pymdp's EFE to prefer STOP regardless of actual
    position (reward collapse when far from food).

    Theoretical underpinning: G(π) = pragmatic_cost(predicted_state) −
    epistemic_value(predicted_uncertainty); argmin_π G(π) selects the mode
    that best reconciles predicted state with preferred outcomes (Eq. 4,
    Friston et al. 2017).

    Hard overrides (applied before EFE scoring):
      1. AVOID — obstacle critically close (d_obs < _D_OBS_CLOSE / 2).
      2. STOP  — food reached and odor confirmed (d_food < _D_FOOD_STOP and
                 c_avg > 0.3).
    """
    d_obs  = controller.body_state.d_obs
    d_food = controller.body_state.d_food
    c_avg  = 0.5 * (controller.body_state.c_left + controller.body_state.c_right)

    if d_obs < _D_OBS_CLOSE * 0.5:
        return BehavioralMode.AVOID
    if d_food < _D_FOOD_STOP and c_avg > 0.3:
        return BehavioralMode.STOP

    # Continuous EFE selection: score all four modes and pick argmin G
    best_mode = min(BehavioralMode, key=lambda m: expected_free_energy(m, controller))
    return best_mode


def mode_to_motor_command(mode: BehavioralMode, controller: ActiveInferenceController) -> dict:
    """
    Map a discrete behavioral mode to a walking motor command dict.

    The command is interpreted by the closed-loop runner to drive flybody
    leg actuators. Keys:
      forward_speed : [0, 1]  normalised forward walking speed
      yaw_rate      : rad/s   positive = left turn
      sidestep      : [-1, 1] lateral stepping speed
      active        : bool    False = halt all stepping (STOP mode)

    Yaw direction for CAST and AVOID is determined from current belief:
      CAST → turn toward higher odor antenna
      AVOID → turn away from nearest obstacle (heuristic: yaw +/-90°)
    """
    delta_c = controller.body_state.delta_c   # positive → left antenna higher
    d_obs   = controller.body_state.d_obs

    if mode == BehavioralMode.SURGE:
        return {"forward_speed": 1.0, "yaw_rate": 0.0, "sidestep": 0.0, "active": True}

    elif mode == BehavioralMode.CAST:
        # Turn toward the side with higher odor to re-acquire the plume
        yaw = 2.0 if delta_c > 0 else -2.0
        return {"forward_speed": 0.3, "yaw_rate": yaw, "sidestep": 0.0, "active": True}

    elif mode == BehavioralMode.AVOID:
        # Sidestep away + yaw away from obstacle; direction is ambiguous without
        # full obstacle geometry so we use a fixed rightward escape heuristic
        return {"forward_speed": 0.0, "yaw_rate": -3.0, "sidestep": -0.5, "active": True}

    elif mode == BehavioralMode.STOP:
        return {"forward_speed": 0.0, "yaw_rate": 0.0, "sidestep": 0.0, "active": False}

    # Fallback (should never reach here)
    return {"forward_speed": 0.0, "yaw_rate": 0.0, "sidestep": 0.0, "active": False}
