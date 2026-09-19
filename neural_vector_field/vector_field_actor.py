"""Neural-vector-field actor for the TD3 command policy."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from neural_vector_field.config import (
    MAX_DRIVE_VEL,
    STEER_LIMITS,
    STATE_ATTRACTOR_STEERING_DECAY_RATE,
    STATE_ATTRACTOR_STEERING_PROPORTION,
    STATE_ATTRACTOR_WHEEL_SPEED_DECAY_RATE,
    STATE_ATTRACTOR_WHEEL_SPEED_PROPORTION,
    WHEEL_RADIUS,
)
from neural_vector_field.control_alignment import ControlAlignmentDiagnostics, control_misalignment_indicator


class NeuralVectorField(nn.Module):
    """Autonomous vector field over the normalized actuator state and error."""

    def __init__(self, vmax_norm: torch.Tensor, hidden: int = 256):
        super().__init__()
        self.register_buffer("vmax", vmax_norm)
        self.net = nn.Sequential(
            nn.Linear(16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, 8), nn.Tanh(),
        )

    def actuator_derivative_components(
        self,
        state_with_error: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded network output and normalized actuator derivative.

        The first returned tensor is the final ``tanh`` network output in
        ``[-1, 1]``.  The second is that output after multiplication by
        ``vmax`` and therefore has units of normalized actuator state per
        second.  Keeping these quantities available separately is useful for
        test-time diagnosis: it distinguishes a small learned vector-field
        output from saturation imposed by the configured rate limits.
        """
        actuator_state, actuator_error = state_with_error.split(8, dim=-1)
        field_input = torch.cat([actuator_state, actuator_error], dim=-1)
        network_tanh_output = self.net(field_input)
        actuator_state_derivative = network_tanh_output * self.vmax
        return network_tanh_output, actuator_state_derivative

    def forward(self, state_with_error: torch.Tensor) -> torch.Tensor:
        """Return the augmented-state derivative without a time-network input.

        The first eight entries are the normalized actuator-state derivative.
        The final eight entries represent the reference error and remain
        constant during one actor integration interval, matching the original
        augmented-state formulation.
        """
        _, actuator_state_derivative = self.actuator_derivative_components(state_with_error)
        zero_error_derivative = torch.zeros_like(actuator_state_derivative)
        return torch.cat([actuator_state_derivative, zero_error_derivative], dim=-1)


class VectorFieldActor(nn.Module):
    """Actor that outputs an 8-D normalized command by integrating the vector field."""

    def __init__(
        self,
        vmax_norm: torch.Tensor,
        step_dt: float = 0.1,
        method: str = "rk2",
        hidden: int = 256,
        device: torch.device | str | None = None,
        integrator: str | None = None,
        integration_substeps: int = 1,
    ):
        super().__init__()
        # `integrator` is a compatibility alias; this actor uses fixed-step midpoint/RK2.
        requested_method = integrator if integrator is not None else method
        normalized_method = str(requested_method).lower()
        if normalized_method not in {"rk2", "midpoint"}:
            raise ValueError(
                "VectorFieldActor uses the fixed-step RK2/midpoint integrator; "
                f"got integrator/method={requested_method!r}."
            )
        self.dt = float(step_dt)
        self.method = "rk2"
        self.integrator = "rk2"
        self.integration_substeps = int(integration_substeps)
        if self.integration_substeps <= 0:
            raise ValueError(
                "integration_substeps must be a positive integer; "
                f"got {integration_substeps!r}."
            )
        self.device = torch.device(device) if device is not None else vmax_norm.device
        # Keep the attribute name `field` so the module structure remains clear.
        self.field = NeuralVectorField(vmax_norm.to(self.device), hidden).to(self.device)

    def _integrate_rk2(self, state_with_error: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the augmented state by fixed-step midpoint/RK2 integration.

        ``dt`` affects only numerical integration. It is never concatenated
        into the neural-network input. When ``integration_substeps > 1``, the
        control interval is divided into equal RK2 substeps; this keeps the
        public actor interface compatible with the existing TD3 constructor.
        """
        state = state_with_error
        sub_dt = float(dt) / float(self.integration_substeps)
        for _ in range(self.integration_substeps):
            k1 = self.field(state)
            midpoint_state = state + 0.5 * sub_dt * k1
            k2 = self.field(midpoint_state)
            state = state + sub_dt * k2
        return state

    def forward_with_diagnostics(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Integrate one control interval and expose the RK2 diagnostic path.

        The normal :meth:`forward` interface deliberately remains unchanged for
        training and deployment.  This method is intended for test-time CSV
        logging.  It reports the bounded network output, rate-limited
        normalized derivatives, RK2 stage states, integration increment and
        output-clamp flags.  No model parameters or control semantics are
        changed by calling it.

        For ``integration_substeps > 1``, the diagnostics retain the first
        substep's ``k1`` quantities and the final substep's ``k2`` quantities.
        The standard testing configuration uses one substep, so these are the
        ordinary RK2 ``k1`` and ``k2`` stages of the current control update.
        """
        observation = observation.to(self.device)
        initial_state_with_error = observation
        state = observation
        sub_dt = float(self.dt) / float(self.integration_substeps)

        first_k1_tanh: torch.Tensor | None = None
        first_k1_derivative: torch.Tensor | None = None
        first_midpoint_state: torch.Tensor | None = None
        final_k2_tanh: torch.Tensor | None = None
        final_k2_derivative: torch.Tensor | None = None

        for substep_index in range(self.integration_substeps):
            k1_tanh, k1_actuator_derivative = self.field.actuator_derivative_components(state)
            k1 = torch.cat([k1_actuator_derivative, torch.zeros_like(k1_actuator_derivative)], dim=-1)
            midpoint_state = state + 0.5 * sub_dt * k1

            k2_tanh, k2_actuator_derivative = self.field.actuator_derivative_components(midpoint_state)
            k2 = torch.cat([k2_actuator_derivative, torch.zeros_like(k2_actuator_derivative)], dim=-1)

            if substep_index == 0:
                first_k1_tanh = k1_tanh
                first_k1_derivative = k1_actuator_derivative
                first_midpoint_state = midpoint_state

            final_k2_tanh = k2_tanh
            final_k2_derivative = k2_actuator_derivative
            state = state + sub_dt * k2

        assert first_k1_tanh is not None
        assert first_k1_derivative is not None
        assert first_midpoint_state is not None
        assert final_k2_tanh is not None
        assert final_k2_derivative is not None

        unclamped_command = state[:, :8]
        normalized_command = unclamped_command.clamp(-1.0, 1.0)
        input_actuator_state = initial_state_with_error[:, :8]
        command_delta = normalized_command - input_actuator_state

        diagnostics = {
            'input_normalized_state': input_actuator_state,
            'input_normalized_error': initial_state_with_error[:, 8:],
            'vmax_normalized_per_s': self.field.vmax.unsqueeze(0).expand_as(input_actuator_state),
            'k1_network_tanh_output': first_k1_tanh,
            'k1_normalized_derivative_per_s': first_k1_derivative,
            'midpoint_normalized_state': first_midpoint_state[:, :8],
            'k2_network_tanh_output': final_k2_tanh,
            'k2_normalized_derivative_per_s': final_k2_derivative,
            'unclamped_normalized_command': unclamped_command,
            'normalized_command': normalized_command,
            'normalized_command_delta': command_delta,
            'normalized_output_low_clamp': (unclamped_command < -1.0),
            'normalized_output_high_clamp': (unclamped_command > 1.0),
            'integration_substep_dt_s': torch.full_like(input_actuator_state, sub_dt),
        }
        return normalized_command, diagnostics

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the 8-D normalized command after one fixed RK2 control step."""
        observation = observation.to(self.device)
        next_state_with_error = self._integrate_rk2(observation, self.dt)
        normalized_command = next_state_with_error[:, :8]
        return normalized_command.clamp(-1.0, 1.0)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _min(values: list[float]) -> float:
        return float(min(values)) if values else 0.0

    @staticmethod
    def _max(values: list[float]) -> float:
        return float(max(values)) if values else 0.0

    @staticmethod
    def _sum_int(values: list[int]) -> int:
        return int(sum(values)) if values else 0

    def rollout(
        self,
        actuator_state: torch.Tensor,
        actuator_error: torch.Tensor,
        dt: float = 0.1,
        k_steps: int = 10,
        k_exp_pos: float = STATE_ATTRACTOR_STEERING_DECAY_RATE,
        k_exp_vel: float = STATE_ATTRACTOR_WHEEL_SPEED_DECAY_RATE,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Roll out the vector field and return regularization terms.

        Returns
        -------
        final_actuator_state:
            Final normalized actuator state after the rollout.
        state_attractor_integral:
            Weighted integrated state-error term used as the state attractor.
        control_misalignment_integral:
            Integrated control-misalignment indicator used as CMS loss.
        """
        actuator_state = actuator_state.to(self.device)
        actuator_error = actuator_error.to(self.device)
        target_actuator_command = actuator_state + actuator_error

        state_attractor_integral = torch.zeros(1, device=self.device)
        control_misalignment_integral = torch.zeros(1, device=self.device)

        sigma_min_means: list[float] = []
        sigma_min_mins: list[float] = []
        sigma_min_maxs: list[float] = []
        sigma_min_p05s: list[float] = []
        sigma_min_p50s: list[float] = []
        sigma_min_p95s: list[float] = []
        sigma_min_near_zero_counts: list[int] = []
        sigma_min_nonfinite_counts: list[int] = []
        sigma_max_means: list[float] = []
        sigma_max_maxs: list[float] = []
        sigma_max_nonfinite_counts: list[int] = []
        singular_value_nonfinite_counts: list[int] = []
        condition_proxy_means: list[float] = []
        condition_proxy_maxs: list[float] = []
        condition_proxy_p95s: list[float] = []
        matrix_abs_means: list[float] = []
        matrix_abs_maxs: list[float] = []
        matrix_fro_means: list[float] = []
        matrix_nonfinite_counts: list[int] = []

        steer_limits = actuator_state.new_tensor(STEER_LIMITS)
        steer_low = steer_limits[:, 0]
        steer_high = steer_limits[:, 1]
        steer_span = steer_high - steer_low

        step_index = torch.arange(k_steps, dtype=torch.float32, device=self.device)
        ratio = step_index / k_steps
        position_weights = torch.exp(-k_exp_pos * (1 - ratio))
        velocity_weights = torch.exp(-k_exp_vel * (1 - ratio))
        position_weights = position_weights / position_weights.sum()
        velocity_weights = velocity_weights / velocity_weights.sum()

        initial_state_with_error = torch.cat([actuator_state, actuator_error], dim=-1)
        state = self._integrate_rk2(initial_state_with_error, dt)[:, :8]

        for k in range(k_steps):
            error = target_actuator_command - state

            steering_error = error[:, :4]
            driving_error = error[:, 4:]
            steering_error_norm = steering_error.pow(2).sum(dim=-1)
            driving_error_norm = driving_error.pow(2).sum(dim=-1)
            state_attractor_integral = state_attractor_integral + (
                STATE_ATTRACTOR_STEERING_PROPORTION * position_weights[k] * steering_error_norm
                + STATE_ATTRACTOR_WHEEL_SPEED_PROPORTION * velocity_weights[k] * driving_error_norm
            ).mean()

            normalized_steering, normalized_driving = state.split(4, dim=-1)
            steering_angle = 0.5 * (normalized_steering + 1.0) * steer_span + steer_low
            wheel_angular_velocity = normalized_driving * (MAX_DRIVE_VEL / WHEEL_RADIUS)
            control_misalignment_result = control_misalignment_indicator(
                steering_angle,
                wheel_angular_velocity,
                row_weights=None,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                control_misalignment, alignment_diag = control_misalignment_result
                assert isinstance(alignment_diag, ControlAlignmentDiagnostics)
                sigma_min_means.append(alignment_diag.sigma_min_mean)
                sigma_min_mins.append(alignment_diag.sigma_min_min)
                sigma_min_maxs.append(alignment_diag.sigma_min_max)
                sigma_min_p05s.append(alignment_diag.sigma_min_p05)
                sigma_min_p50s.append(alignment_diag.sigma_min_p50)
                sigma_min_p95s.append(alignment_diag.sigma_min_p95)
                sigma_min_near_zero_counts.append(alignment_diag.sigma_min_near_zero_count)
                sigma_min_nonfinite_counts.append(alignment_diag.sigma_min_nonfinite_count)
                sigma_max_means.append(alignment_diag.sigma_max_mean)
                sigma_max_maxs.append(alignment_diag.sigma_max_max)
                sigma_max_nonfinite_counts.append(alignment_diag.sigma_max_nonfinite_count)
                singular_value_nonfinite_counts.append(alignment_diag.singular_value_nonfinite_count)
                condition_proxy_means.append(alignment_diag.condition_proxy_mean)
                condition_proxy_maxs.append(alignment_diag.condition_proxy_max)
                condition_proxy_p95s.append(alignment_diag.condition_proxy_p95)
                matrix_abs_means.append(alignment_diag.matrix_abs_mean)
                matrix_abs_maxs.append(alignment_diag.matrix_abs_max)
                matrix_fro_means.append(alignment_diag.matrix_fro_mean)
                matrix_nonfinite_counts.append(alignment_diag.matrix_nonfinite_count)
            else:
                control_misalignment = control_misalignment_result
            control_misalignment_integral = control_misalignment_integral + control_misalignment

            state_with_error = torch.cat([state, error], dim=-1)
            state = self._integrate_rk2(state_with_error, dt)[:, :8]

        if not return_diagnostics:
            return state, state_attractor_integral, control_misalignment_integral

        diagnostics = {
            "cms_sigma_min_mean": self._mean(sigma_min_means),
            "cms_sigma_min_min": self._min(sigma_min_mins),
            "cms_sigma_min_max": self._max(sigma_min_maxs),
            "cms_sigma_min_p05_mean": self._mean(sigma_min_p05s),
            "cms_sigma_min_p50_mean": self._mean(sigma_min_p50s),
            "cms_sigma_min_p95_mean": self._mean(sigma_min_p95s),
            "cms_sigma_min_near_zero_count": self._sum_int(sigma_min_near_zero_counts),
            "cms_sigma_min_nonfinite_count": self._sum_int(sigma_min_nonfinite_counts),
            "cms_sigma_max_mean": self._mean(sigma_max_means),
            "cms_sigma_max_max": self._max(sigma_max_maxs),
            "cms_sigma_max_nonfinite_count": self._sum_int(sigma_max_nonfinite_counts),
            "cms_singular_value_nonfinite_count": self._sum_int(singular_value_nonfinite_counts),
            "cms_condition_proxy_mean": self._mean(condition_proxy_means),
            "cms_condition_proxy_max": self._max(condition_proxy_maxs),
            "cms_condition_proxy_p95_mean": self._mean(condition_proxy_p95s),
            "cms_matrix_abs_mean": self._mean(matrix_abs_means),
            "cms_matrix_abs_max": self._max(matrix_abs_maxs),
            "cms_matrix_fro_mean": self._mean(matrix_fro_means),
            "cms_matrix_nonfinite_count": self._sum_int(matrix_nonfinite_counts),
        }
        return state, state_attractor_integral, control_misalignment_integral, diagnostics
