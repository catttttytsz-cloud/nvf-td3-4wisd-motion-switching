"""Control-alignment / misalignment indicator and diagnostics.

Two matrix constructions are available:

``legacy``
    The submitted physical-unit homogeneous matrix.

``normalized``
    A dimensionless matrix for condition assessment. Dimensional entries are
    divided by platform-dependent maximum scales before SVD. The zero-alignment
    condition is preserved, while the numerical scale of the singular-value
    regularizer becomes explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_vector_field.config import (
    CONDITION_PROXY_EPS,
    CONTROL_ALIGNMENT_LENGTH_SCALE,
    CONTROL_ALIGNMENT_MATRIX_MODE,
    CONTROL_ALIGNMENT_SPEED_SCALE,
    SVD_NEAR_ZERO_THRESHOLD,
    WHEEL_POS,
    WHEEL_RADIUS,
)

# The submitted implementation computed candidate physical scaling constants,
# but the effective row weights used in training were all ones. Kept for legacy
# mode and optional compatibility.
ROW_WEIGHTS = torch.tensor([1.0, 1.0] * 4, dtype=torch.float32)


@dataclass(frozen=True)
class ControlAlignmentDiagnostics:
    """Detached scalar diagnostics returned by the alignment indicator."""

    matrix_mode: str
    sigma_min_mean: float
    sigma_min_min: float
    sigma_min_max: float
    sigma_min_p05: float
    sigma_min_p50: float
    sigma_min_p95: float
    sigma_min_near_zero_count: int
    sigma_min_nonfinite_count: int
    sigma_max_mean: float
    sigma_max_max: float
    sigma_max_nonfinite_count: int
    singular_value_nonfinite_count: int
    condition_proxy_mean: float
    condition_proxy_max: float
    condition_proxy_p95: float
    matrix_abs_mean: float
    matrix_abs_max: float
    matrix_fro_mean: float
    matrix_nonfinite_count: int
    length_scale: float
    speed_scale: float


def _safe_quantile(values: torch.Tensor, q: float) -> float:
    """Return a quantile of finite values only; return 0 if none exist."""
    finite_values = values.detach()[torch.isfinite(values.detach())]
    if finite_values.numel() == 0:
        return 0.0
    return float(torch.quantile(finite_values.cpu(), q).item())


def _finite_stat(values: torch.Tensor, reducer: str, default: float = 0.0) -> float:
    finite_values = values.detach()[torch.isfinite(values.detach())]
    if finite_values.numel() == 0:
        return default
    if reducer == "mean":
        return float(finite_values.mean().cpu())
    if reducer == "min":
        return float(finite_values.min().cpu())
    if reducer == "max":
        return float(finite_values.max().cpu())
    raise ValueError(f"Unknown reducer: {reducer}")


def _build_alignment_matrix(
    steering_angle: torch.Tensor,
    wheel_angular_velocity: torch.Tensor,
    *,
    matrix_mode: str,
) -> torch.Tensor:
    """Build the batched homogeneous constraint matrix.

    Parameters
    ----------
    steering_angle:
        Tensor of shape ``(B, 4)`` in radians, ordered as rear-left,
        front-left, rear-right, front-right.
    wheel_angular_velocity:
        Tensor of shape ``(B, 4)``. In the current rollout this is the effective
        wheel angular quantity passed to the SVD loss. The dimensional entry
        entering the matrix is ``WHEEL_RADIUS * omega``.
    matrix_mode:
        ``legacy`` keeps physical-unit position and speed entries. ``normalized``
        divides position and wheel-linear-speed entries by their maximum scales.
    """
    batch_size = steering_angle.shape[0]
    device = steering_angle.device
    dtype = steering_angle.dtype

    if matrix_mode not in {"legacy", "normalized"}:
        raise ValueError(f"Unknown control-alignment matrix mode: {matrix_mode!r}")

    cos_d = torch.cos(steering_angle)
    sin_d = torch.sin(steering_angle)
    tangent = torch.stack([cos_d, sin_d], dim=-1)
    normal = torch.stack([-sin_d, cos_d], dim=-1)

    wheel_pos = torch.as_tensor(WHEEL_POS, device=device, dtype=dtype).unsqueeze(0)
    wheel_pos = wheel_pos.expand(batch_size, -1, -1)
    wheel_linear_speed = WHEEL_RADIUS * wheel_angular_velocity

    if matrix_mode == "normalized":
        length_scale = torch.as_tensor(CONTROL_ALIGNMENT_LENGTH_SCALE, device=device, dtype=dtype)
        speed_scale = torch.as_tensor(CONTROL_ALIGNMENT_SPEED_SCALE, device=device, dtype=dtype)
        # Defensive guards: scales are constants from config and should be > 0.
        wheel_pos = wheel_pos / length_scale.clamp_min(torch.finfo(dtype).eps)
        wheel_linear_speed = wheel_linear_speed / speed_scale.clamp_min(torch.finfo(dtype).eps)

    x_pos = wheel_pos[..., 0]
    y_pos = wheel_pos[..., 1]

    normal_cross = (normal[..., 0] * y_pos - normal[..., 1] * x_pos).unsqueeze(-1)
    tangent_cross = (tangent[..., 0] * y_pos - tangent[..., 1] * x_pos).unsqueeze(-1)

    matrix = torch.zeros(batch_size, 8, 4, device=device, dtype=dtype)
    for wheel_id in range(4):
        matrix[:, 2 * wheel_id, 0:2] = normal[:, wheel_id]
        matrix[:, 2 * wheel_id, 2:3] = normal_cross[:, wheel_id]
        matrix[:, 2 * wheel_id, 3] = 0.0

        matrix[:, 2 * wheel_id + 1, 0:2] = tangent[:, wheel_id]
        matrix[:, 2 * wheel_id + 1, 2:3] = tangent_cross[:, wheel_id]
        matrix[:, 2 * wheel_id + 1, 3] = -wheel_linear_speed[:, wheel_id]

    return matrix


def control_misalignment_indicator(
    steering_angle: torch.Tensor,
    wheel_angular_velocity: torch.Tensor,
    row_weights: torch.Tensor | None = None,
    matrix_mode: str | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ControlAlignmentDiagnostics]:
    """Return the batch-mean minimum singular value of the alignment matrix."""
    mode = CONTROL_ALIGNMENT_MATRIX_MODE if matrix_mode is None else matrix_mode
    matrix = _build_alignment_matrix(
        steering_angle,
        wheel_angular_velocity,
        matrix_mode=mode,
    )

    weights = ROW_WEIGHTS if row_weights is None else row_weights
    matrix = matrix * weights.to(device=matrix.device, dtype=matrix.dtype).view(1, 8, 1)

    singular_values = torch.linalg.svdvals(matrix)
    sigma_min = singular_values[:, -1]
    loss = sigma_min.mean()

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        sigma_max = singular_values[:, 0]
        finite_sigma_min = torch.where(
            torch.isfinite(sigma_min),
            sigma_min,
            torch.zeros_like(sigma_min),
        )
        condition_proxy = sigma_max / finite_sigma_min.clamp_min(CONDITION_PROXY_EPS)
        matrix_fro = torch.linalg.vector_norm(matrix.reshape(matrix.shape[0], -1), dim=-1)
        diagnostics = ControlAlignmentDiagnostics(
            matrix_mode=mode,
            sigma_min_mean=_finite_stat(sigma_min, "mean"),
            sigma_min_min=_finite_stat(sigma_min, "min"),
            sigma_min_max=_finite_stat(sigma_min, "max"),
            sigma_min_p05=_safe_quantile(sigma_min, 0.05),
            sigma_min_p50=_safe_quantile(sigma_min, 0.50),
            sigma_min_p95=_safe_quantile(sigma_min, 0.95),
            sigma_min_near_zero_count=int((finite_sigma_min.abs() < SVD_NEAR_ZERO_THRESHOLD).sum().cpu()),
            sigma_min_nonfinite_count=int((~torch.isfinite(sigma_min)).sum().cpu()),
            sigma_max_mean=_finite_stat(sigma_max, "mean"),
            sigma_max_max=_finite_stat(sigma_max, "max"),
            sigma_max_nonfinite_count=int((~torch.isfinite(sigma_max)).sum().cpu()),
            singular_value_nonfinite_count=int((~torch.isfinite(singular_values)).sum().cpu()),
            condition_proxy_mean=_finite_stat(condition_proxy, "mean"),
            condition_proxy_max=_finite_stat(condition_proxy, "max"),
            condition_proxy_p95=_safe_quantile(condition_proxy, 0.95),
            matrix_abs_mean=_finite_stat(matrix.abs(), "mean"),
            matrix_abs_max=_finite_stat(matrix.abs(), "max"),
            matrix_fro_mean=_finite_stat(matrix_fro, "mean"),
            matrix_nonfinite_count=int((~torch.isfinite(matrix)).sum().cpu()),
            length_scale=float(CONTROL_ALIGNMENT_LENGTH_SCALE),
            speed_scale=float(CONTROL_ALIGNMENT_SPEED_SCALE),
        )
    return loss, diagnostics
