"""Target-twist samplers for the five paper-specified modes: x, y, z, xy, xz."""

from __future__ import annotations

import random

from neural_vector_field.config import MAX_ANGULAR_Z, MAX_LINEAR_X, MAX_LINEAR_Y


def _near_zero() -> float:
    return random.uniform(-0.0001, 0.0001)


def _biased_uniform(
    max_val: float,
    band_probs: tuple[float, float, float] = (0.15, 0.35, 0.50),
    low_cut: float = 0.30,
    mid_cut: float = 0.70,
) -> float:
    assert 0.0 < low_cut < mid_cut < 1.0
    assert abs(sum(band_probs) - 1.0) < 1e-6

    sample = random.random()
    if sample < band_probs[0]:
        magnitude = random.uniform(0.0, low_cut * max_val)
    elif sample < band_probs[0] + band_probs[1]:
        magnitude = random.uniform(low_cut * max_val, mid_cut * max_val)
    else:
        magnitude = random.uniform(mid_cut * max_val, max_val)
    return magnitude if random.random() < 0.5 else -magnitude


def _sample_mode(mode_probabilities: dict[str, float]) -> str:
    cumulative = 0.0
    sample = random.random()
    last_mode = next(reversed(mode_probabilities))
    for mode, probability in mode_probabilities.items():
        cumulative += probability
        if sample < cumulative:
            return mode
    return last_mode


def sample_target_twist() -> list[float]:
    """Sample a target base twist from the nominal training distribution."""
    mode = _sample_mode({
        "x": 0.10,
        "y": 0.36,
        "z": 0.20,
        "xy": 0.17,
        "xz": 0.17,
    })

    if mode == "x":
        return [_biased_uniform(MAX_LINEAR_X), _near_zero(), _near_zero()]

    if mode == "y":
        return [_near_zero(), _biased_uniform(MAX_LINEAR_Y), _near_zero()]

    if mode == "z":
        return [_near_zero(), _near_zero(), _biased_uniform(MAX_ANGULAR_Z)]

    if mode == "xy":
        threshold = 0.05
        x_vel = _biased_uniform(MAX_LINEAR_X)
        while abs(x_vel) < threshold:
            x_vel = _biased_uniform(MAX_LINEAR_X)
        max_y = min(abs(x_vel), MAX_LINEAR_Y)
        y_vel = _biased_uniform(max_y)
        if abs(y_vel) >= abs(x_vel):
            y_vel = abs(x_vel) * 0.9 * (1 if random.random() < 0.5 else -1)
        return [x_vel, y_vel, _near_zero()]

    if mode == "xz":
        threshold = 0.05
        x_vel = _biased_uniform(MAX_LINEAR_X)
        while abs(x_vel) < threshold:
            x_vel = _biased_uniform(MAX_LINEAR_X)
        return [x_vel, _near_zero(), _biased_uniform(MAX_ANGULAR_Z)]

    raise ValueError(f"Unknown mode: {mode}")


def sample_abrupt_target_twist() -> list[float]:
    """Sample a target twist that avoids repeating the immediately previous mode."""
    mode_probabilities = {
        "x": 0.0,
        "y": 0.50,
        "z": 0.25,
        "xy": 0.25,
        "xz": 0.0,
    }
    if not hasattr(sample_abrupt_target_twist, "last_mode"):
        sample_abrupt_target_twist.last_mode = None

    while True:
        mode = _sample_mode(mode_probabilities)
        if mode != sample_abrupt_target_twist.last_mode:
            break
    sample_abrupt_target_twist.last_mode = mode

    if mode == "x":
        return [_biased_uniform(MAX_LINEAR_X, (0.0, 0.35, 0.65)), _near_zero(), _near_zero()]
    if mode == "y":
        return [_near_zero(), _biased_uniform(MAX_LINEAR_Y, (0.0, 0.35, 0.65)), _near_zero()]
    if mode == "z":
        return [_near_zero(), _near_zero(), _biased_uniform(MAX_ANGULAR_Z, (0.0, 0.35, 0.65))]
    if mode == "xy":
        threshold = 0.05
        x_vel = _biased_uniform(MAX_LINEAR_X, (0.0, 0.35, 0.65))
        while abs(x_vel) < threshold:
            x_vel = _biased_uniform(MAX_LINEAR_X, (0.0, 0.35, 0.65))
        max_y = min(abs(x_vel), MAX_LINEAR_Y)
        y_vel = _biased_uniform(max_y, (0.0, 0.35, 0.65))
        if abs(y_vel) >= abs(x_vel):
            y_vel = abs(x_vel) * 0.9 * (1 if random.random() < 0.5 else -1)
        return [x_vel, y_vel, _near_zero()]
    if mode == "xz":
        threshold = 0.05
        x_vel = _biased_uniform(MAX_LINEAR_X, (0.0, 0.35, 0.65))
        while abs(x_vel) < threshold:
            x_vel = _biased_uniform(MAX_LINEAR_X, (0.0, 0.35, 0.65))
        return [x_vel, _near_zero(), _biased_uniform(MAX_ANGULAR_Z, (0.0, 0.35, 0.65))]

    raise ValueError(f"Unknown mode: {mode}")

# Backward-compatible aliases.
generate_command = sample_target_twist
generate_command_abrupt = sample_abrupt_target_twist
