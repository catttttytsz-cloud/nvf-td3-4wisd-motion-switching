"""TD3 agent with neural-vector-field actor and rollout regularizers."""

from __future__ import annotations

from typing import Tuple
import math
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_vector_field.config import (
    ACTOR_GRADIENT_CLIP_NORM,
    ACTOR_LR,
    ACTOR_LR_DECAY_END_EPISODE,
    ACTOR_LR_DECAY_START_EPISODE,
    ACTOR_LR_MIN,
    BATCH_SIZE,
    BUFFER_SIZE,
    COMMAND_DIM,
    CRITIC_LR,
    CRITIC_LR_DECAY_END_EPISODE,
    CRITIC_LR_HOLD_END_EPISODE,
    CRITIC_LR_INIT,
    CRITIC_LR_MIN,
    CRITIC_LR_PEAK,
    CRITIC_LR_WARMUP_END_EPISODE,
    ENABLE_ACTOR_COMPONENT_GRADIENT_LOGGING,
    ENABLE_ACTOR_GRADIENT_CLIPPING,
    ENABLE_ACTOR_LR_COSINE_DECAY,
    ENABLE_CRITIC_LR_SCHEDULE,
    ENABLE_PARAMETER_GRADIENT_LOGGING,
    ACTOR_DIAGNOSTIC_INTERVAL_UPDATES,
    ACTOR_INTEGRATION_SUBSTEPS,
    EXPLORATION_NOISE_DECAY_PER_EPISODE,
    EXPLORATION_NOISE_STD,
    GAMMA,
    LAMBDA_CONTROL_MISALIGNMENT,
    LAMBDA_STATE_ATTRACTOR,
    MIN_EXPLORATION_NOISE_STD,
    POLICY_UPDATE_FREQ,
    ROLLOUT_HORIZON,
    ROLLOUT_STEP_SIZE,
    STATE_DIM,
    TARGET_POLICY_NOISE_CLIP,
    TARGET_POLICY_NOISE_STD,
    TAU,
)
from neural_vector_field.vector_field_actor import VectorFieldActor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(sizes: list[int], act=nn.ELU, out_act=nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        layers.append(act() if idx < len(sizes) - 2 else out_act())
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = _mlp([STATE_DIM + COMMAND_DIM, 512, 256, 1])

    def forward(self, observation: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([observation, command], dim=-1))


class ReplayBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.obs = np.empty((BUFFER_SIZE, STATE_DIM), np.float32)
        self.command = np.empty((BUFFER_SIZE, COMMAND_DIM), np.float32)
        self.reward = np.empty((BUFFER_SIZE, 1), np.float32)
        self.next_obs = np.empty((BUFFER_SIZE, STATE_DIM), np.float32)
        self.done = np.empty((BUFFER_SIZE, 1), np.float32)
        self.ptr = 0
        self.full = False

    def push(self, obs, command, reward, next_obs, done) -> None:
        with self.lock:
            self.obs[self.ptr] = obs
            self.command[self.ptr] = command
            self.reward[self.ptr] = reward
            self.next_obs[self.ptr] = next_obs
            self.done[self.ptr] = done
            self.ptr = (self.ptr + 1) % BUFFER_SIZE
            if self.ptr == 0:
                self.full = True

    def size(self) -> int:
        with self.lock:
            return BUFFER_SIZE if self.full else self.ptr

    def sample(self, batch: int = BATCH_SIZE) -> Tuple[torch.Tensor, ...]:
        with self.lock:
            size = BUFFER_SIZE if self.full else self.ptr
            if size < batch:
                raise ValueError(f"ReplayBuffer sample needs batch={batch}, but only has size={size}.")
            idx = np.random.choice(size, batch, replace=False)
            obs = self.obs[idx].copy()
            command = self.command[idx].copy()
            reward = self.reward[idx].copy()
            next_obs = self.next_obs[idx].copy()
            done = self.done[idx].copy()
        return (
            torch.as_tensor(obs, device=DEVICE),
            torch.as_tensor(command, device=DEVICE),
            torch.as_tensor(reward, device=DEVICE),
            torch.as_tensor(next_obs, device=DEVICE),
            torch.as_tensor(done, device=DEVICE),
        )


class TD3VectorFieldAgent:
    def __init__(
        self,
        command_low: np.ndarray,
        command_high: np.ndarray,
        vmax_norm: np.ndarray,
        env_dt: float,
    ):
        vmax_tensor = torch.as_tensor(vmax_norm, device=DEVICE)
        self.actor = VectorFieldActor(vmax_tensor, step_dt=env_dt, device=DEVICE, integration_substeps=ACTOR_INTEGRATION_SUBSTEPS).to(DEVICE)
        self.actor_target = VectorFieldActor(vmax_tensor, step_dt=env_dt, device=DEVICE, integration_substeps=ACTOR_INTEGRATION_SUBSTEPS).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = Critic().to(DEVICE)
        self.q2 = Critic().to(DEVICE)
        self.q1_target = Critic().to(DEVICE)
        self.q2_target = Critic().to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=CRITIC_LR,
        )
        self.replay_buffer = ReplayBuffer()
        self.actor_lock = threading.RLock()
        self.update_lock = threading.RLock()

        self.command_low = command_low
        self.command_high = command_high
        self.command_low_t = torch.as_tensor(command_low, device=DEVICE)
        self.command_high_t = torch.as_tensor(command_high, device=DEVICE)
        self.total_update_step = 0
        self.actor_update_step = 0
        self.current_actor_lr = float(ACTOR_LR)
        self.current_critic_lr = float(CRITIC_LR)
        self.last_update_info: dict[str, float | int | bool] = {}

    @torch.no_grad()
    def select_command(self, observation_np: np.ndarray, episode: int) -> np.ndarray:
        
        
        noise_std = max(
            MIN_EXPLORATION_NOISE_STD,
            EXPLORATION_NOISE_STD - episode * EXPLORATION_NOISE_DECAY_PER_EPISODE,
        )

        observation = torch.as_tensor(observation_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with self.actor_lock:
            command = self.actor(observation).cpu().numpy()[0]
        if noise_std > 0.0:
            command += noise_std * np.random.randn(COMMAND_DIM)
        return np.clip(command, self.command_low, self.command_high)

    def push(self, *args) -> None:
        self.replay_buffer.push(*args)

    def size(self) -> int:
        return self.replay_buffer.size()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - TAU)
            target_param.data.add_(TAU * source_param.data)

    @staticmethod
    def _tensor_to_float(value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _vector_stats(vector: torch.Tensor) -> dict[str, float | int | bool]:
        if vector.numel() == 0:
            return {
                "grad_norm": 0.0,
                "grad_max_abs": 0.0,
                "grad_mean_abs": 0.0,
                "grad_element_count": 0,
                "grad_nonfinite_count": 0,
                "grad_is_finite": True,
            }
        vector_detached = vector.detach()
        finite_mask = torch.isfinite(vector_detached)
        nonfinite_count = int((~finite_mask).sum().detach().cpu())
        safe_vector = torch.where(finite_mask, vector_detached, torch.zeros_like(vector_detached))
        return {
            "grad_norm": float(torch.linalg.vector_norm(safe_vector).detach().cpu()),
            "grad_max_abs": float(safe_vector.abs().max().detach().cpu()),
            "grad_mean_abs": float(safe_vector.abs().mean().detach().cpu()),
            "grad_element_count": int(safe_vector.numel()),
            "grad_nonfinite_count": nonfinite_count,
            "grad_is_finite": nonfinite_count == 0,
        }

    def _flatten_gradients(self, gradients) -> torch.Tensor:
        flat_parts = []
        params = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        for parameter, grad in zip(params, gradients):
            if grad is None:
                flat_parts.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1))
            else:
                flat_parts.append(grad.reshape(-1))
        if not flat_parts:
            return torch.empty(0, device=DEVICE)
        return torch.cat(flat_parts)

    def _actor_gradient_vector(self) -> torch.Tensor:
        params = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        return self._flatten_gradients([parameter.grad for parameter in params])

    def _component_gradient_vector(self, loss: torch.Tensor) -> torch.Tensor:
        params = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        return self._flatten_gradients(gradients)

    def _prefixed_stats(self, prefix: str, vector: torch.Tensor) -> dict[str, float | int | bool]:
        stats = self._vector_stats(vector)
        return {f"{prefix}_{key}": value for key, value in stats.items()}

    @staticmethod
    def _cosine(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
        if vector_a.numel() == 0 or vector_b.numel() == 0:
            return 0.0
        finite_a = torch.where(torch.isfinite(vector_a), vector_a, torch.zeros_like(vector_a))
        finite_b = torch.where(torch.isfinite(vector_b), vector_b, torch.zeros_like(vector_b))
        denom = torch.linalg.vector_norm(finite_a) * torch.linalg.vector_norm(finite_b)
        if float(denom.detach().cpu()) <= 1e-12:
            return 0.0
        return float((torch.dot(finite_a, finite_b) / denom).detach().cpu())

    def _actor_parameter_gradient_stats(self) -> dict[str, float | int | bool]:
        stats: dict[str, float | int | bool] = {}
        for name, parameter in self.actor.named_parameters():
            if parameter.grad is None:
                continue
            safe_name = name.replace(".", "_")
            grad_stats = self._vector_stats(parameter.grad.reshape(-1))
            stats[f"param_grad_{safe_name}_norm"] = float(grad_stats["grad_norm"])
            stats[f"param_grad_{safe_name}_max_abs"] = float(grad_stats["grad_max_abs"])
            stats[f"param_grad_{safe_name}_mean_abs"] = float(grad_stats["grad_mean_abs"])
            stats[f"param_grad_{safe_name}_nonfinite_count"] = int(grad_stats["grad_nonfinite_count"])
        return stats

    @staticmethod
    def actor_lr_for_episode(episode: int | None) -> float:
        """Return the actor learning rate for the current episode.

        Episode-based decay remains aligned with the training phase even when
        the background learner drops queued optimizer updates.
        """
        if not ENABLE_ACTOR_LR_COSINE_DECAY:
            return float(ACTOR_LR)
        if episode is None:
            return float(ACTOR_LR)

        episode_i = int(episode)
        start = int(ACTOR_LR_DECAY_START_EPISODE)
        end = int(ACTOR_LR_DECAY_END_EPISODE)
        lr_max = float(ACTOR_LR)
        lr_min = float(ACTOR_LR_MIN)

        if end <= start:
            return lr_min if episode_i >= start else lr_max
        if episode_i < start:
            return lr_max
        if episode_i >= end:
            return lr_min

        progress = (episode_i - start) / max(float(end - start), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine

    def _set_actor_lr(self, lr: float) -> None:
        lr_f = float(lr)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr_f
        self.current_actor_lr = lr_f

    @staticmethod
    def critic_lr_for_episode(episode: int | None) -> float:
        """Return the warmup-hold-decay critic learning rate for an episode."""
        if not ENABLE_CRITIC_LR_SCHEDULE:
            return float(CRITIC_LR)
        if episode is None:
            return float(CRITIC_LR)

        episode_i = int(episode)
        warmup_end = int(CRITIC_LR_WARMUP_END_EPISODE)
        hold_end = int(CRITIC_LR_HOLD_END_EPISODE)
        decay_end = int(CRITIC_LR_DECAY_END_EPISODE)
        lr_init = float(CRITIC_LR_INIT)
        lr_peak = float(CRITIC_LR_PEAK)
        lr_min = float(CRITIC_LR_MIN)

        if episode_i < 0:
            return lr_init
        if warmup_end > 0 and episode_i < warmup_end:
            progress = episode_i / max(float(warmup_end), 1.0)
            return lr_init + (lr_peak - lr_init) * progress
        if episode_i < hold_end or decay_end <= hold_end:
            return lr_peak
        if episode_i >= decay_end:
            return lr_min

        progress = (episode_i - hold_end) / max(float(decay_end - hold_end), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (lr_peak - lr_min) * cosine

    def _set_critic_lr(self, lr: float) -> None:
        lr_f = float(lr)
        for group in self.critic_optimizer.param_groups:
            group["lr"] = lr_f
        self.current_critic_lr = lr_f

    def update(self, episode: int | None = None, force_actor_diagnostics: bool | None = None) -> float:
        with self.update_lock:
            return self._update_impl(episode=episode, force_actor_diagnostics=force_actor_diagnostics)

    def _update_impl(self, episode: int | None = None, force_actor_diagnostics: bool | None = None) -> float:
        self.total_update_step += 1
        actor_lr = self.actor_lr_for_episode(episode)
        critic_lr = self.critic_lr_for_episode(episode)
        self._set_actor_lr(actor_lr)
        self._set_critic_lr(critic_lr)
        obs, command, reward, next_obs, done = self.replay_buffer.sample()

        with torch.no_grad():
            target_noise = TARGET_POLICY_NOISE_STD * torch.randn_like(command)
            target_noise = target_noise.clamp(-TARGET_POLICY_NOISE_CLIP, TARGET_POLICY_NOISE_CLIP)
            next_command = self.actor_target(next_obs) + target_noise
            next_command = torch.max(torch.min(next_command, self.command_high_t), self.command_low_t)
            target_q = torch.min(
                self.q1_target(next_obs, next_command),
                self.q2_target(next_obs, next_command),
            )
            td_target = reward + GAMMA * (1.0 - done) * target_q

        q1_current = self.q1(obs, command)
        q2_current = self.q2(obs, command)
        critic_loss = F.mse_loss(q1_current, td_target) + F.mse_loss(q2_current, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            td_error1 = (q1_current - td_target).abs()
            td_error2 = (q2_current - td_target).abs()
            q_gap_current = (q1_current - q2_current).abs()
        self.last_update_info = {
            "actor_updated": False,
            "critic_loss": self._tensor_to_float(critic_loss),
            "q_value_mean": self._tensor_to_float(q1_current.mean()),
            "q1_current_mean": self._tensor_to_float(q1_current.mean()),
            "q2_current_mean": self._tensor_to_float(q2_current.mean()),
            "q_current_gap_mean": self._tensor_to_float(q_gap_current.mean()),
            "target_q_mean": self._tensor_to_float(target_q.mean()),
            "td_target_mean": self._tensor_to_float(td_target.mean()),
            "td_error1_abs_mean": self._tensor_to_float(td_error1.mean()),
            "td_error2_abs_mean": self._tensor_to_float(td_error2.mean()),
            "sample_reward_mean": self._tensor_to_float(reward.mean()),
            "actor_update_index": self.actor_update_step,
            "total_update_step": self.total_update_step,
        }

        if self.total_update_step % POLICY_UPDATE_FREQ == 0:
            with self.actor_lock:
                self.actor_update_step += 1
                if force_actor_diagnostics is None:
                    do_actor_diagnostics = (
                        ACTOR_DIAGNOSTIC_INTERVAL_UPDATES > 0
                        and self.actor_update_step % ACTOR_DIAGNOSTIC_INTERVAL_UPDATES == 0
                    )
                else:
                    do_actor_diagnostics = bool(force_actor_diagnostics)
                actor_command = self.actor(obs)
                current_actuator_state = obs[:, :8]
                actuator_state_error = obs[:, 8:]
                rollout_result = self.actor.rollout(
                    current_actuator_state,
                    actuator_state_error,
                    dt=ROLLOUT_STEP_SIZE,
                    k_steps=ROLLOUT_HORIZON,
                    return_diagnostics=do_actor_diagnostics,
                )
                if do_actor_diagnostics:
                    _, state_attractor_loss, control_misalignment_loss, rollout_diagnostics = rollout_result
                else:
                    _, state_attractor_loss, control_misalignment_loss = rollout_result
                    rollout_diagnostics = {}

                q1_pi = self.q1(obs, actor_command)
                q2_pi = self.q2(obs, actor_command)
                q_term = -q1_pi.mean()
                state_attractor_term = LAMBDA_STATE_ATTRACTOR * state_attractor_loss
                control_misalignment_term = LAMBDA_CONTROL_MISALIGNMENT * control_misalignment_loss
                actor_loss = q_term + state_attractor_term + control_misalignment_term

                component_gradient_stats: dict[str, float | int | bool] = {}
                component_cosines: dict[str, float] = {}
                if do_actor_diagnostics and ENABLE_ACTOR_COMPONENT_GRADIENT_LOGGING:
                    actor_loss_grad = self._component_gradient_vector(actor_loss)
                    q_grad = self._component_gradient_vector(q_term)
                    state_attractor_grad = self._component_gradient_vector(state_attractor_term)
                    control_misalignment_grad = self._component_gradient_vector(control_misalignment_term)
                    component_gradient_stats.update(self._prefixed_stats("grad_actor_loss", actor_loss_grad))
                    component_gradient_stats.update(self._prefixed_stats("grad_q_term", q_grad))
                    component_gradient_stats.update(self._prefixed_stats("grad_state_attractor_term", state_attractor_grad))
                    component_gradient_stats.update(self._prefixed_stats("grad_control_misalignment_term", control_misalignment_grad))
                    component_cosines.update({
                        "cos_q_state_attractor": self._cosine(q_grad, state_attractor_grad),
                        "cos_q_control_misalignment": self._cosine(q_grad, control_misalignment_grad),
                        "cos_state_attractor_control_misalignment": self._cosine(state_attractor_grad, control_misalignment_grad),
                        "cos_q_actor_loss": self._cosine(q_grad, actor_loss_grad),
                        "cos_state_attractor_actor_loss": self._cosine(state_attractor_grad, actor_loss_grad),
                        "cos_control_misalignment_actor_loss": self._cosine(control_misalignment_grad, actor_loss_grad),
                    })

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_vector = self._actor_gradient_vector()
                grad_stats_before_clip = self._vector_stats(actor_grad_vector)
                actor_grad_norm = float(grad_stats_before_clip["grad_norm"])
                actor_grad_would_clip = bool(actor_grad_norm > ACTOR_GRADIENT_CLIP_NORM)
                actor_grad_clipped = False
                if ENABLE_ACTOR_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        max_norm=ACTOR_GRADIENT_CLIP_NORM,
                        error_if_nonfinite=False,
                    )
                    actor_grad_clipped = actor_grad_would_clip
                grad_stats_after_clip = self._vector_stats(self._actor_gradient_vector())
                parameter_gradient_stats = (
                    self._actor_parameter_gradient_stats() if ENABLE_PARAMETER_GRADIENT_LOGGING else {}
                )
                self.actor_optimizer.step()

                with torch.no_grad():
                    q_pi_gap = (q1_pi - q2_pi).abs()
                self.last_update_info = {
                    "actor_updated": True,
                    "critic_loss": self._tensor_to_float(critic_loss),
                    "q_value_mean": self._tensor_to_float(q1_current.mean()),
                    "q1_current_mean": self._tensor_to_float(q1_current.mean()),
                    "q2_current_mean": self._tensor_to_float(q2_current.mean()),
                    "q_current_gap_mean": self._tensor_to_float(q_gap_current.mean()),
                    "target_q_mean": self._tensor_to_float(target_q.mean()),
                    "td_target_mean": self._tensor_to_float(td_target.mean()),
                    "td_error1_abs_mean": self._tensor_to_float(td_error1.mean()),
                    "td_error2_abs_mean": self._tensor_to_float(td_error2.mean()),
                    "sample_reward_mean": self._tensor_to_float(reward.mean()),
                    "actor_update_index": self.actor_update_step,
                    "total_update_step": self.total_update_step,
                    "actor_diagnostics_enabled": bool(do_actor_diagnostics),
                    "actor_loss": self._tensor_to_float(actor_loss),
                    "q_term": self._tensor_to_float(q_term),
                    "state_attractor_loss": self._tensor_to_float(state_attractor_loss),
                    "state_attractor_term": self._tensor_to_float(state_attractor_term),
                    "control_misalignment_loss": self._tensor_to_float(control_misalignment_loss),
                    "control_misalignment_term": self._tensor_to_float(control_misalignment_term),
                    "q1_pi_mean": self._tensor_to_float(q1_pi.mean()),
                    "q2_pi_mean": self._tensor_to_float(q2_pi.mean()),
                    "q_pi_gap_mean": self._tensor_to_float(q_pi_gap.mean()),
                    "actor_gradient_clipping_enabled": ENABLE_ACTOR_GRADIENT_CLIPPING,
                    "actor_gradient_clip_norm": ACTOR_GRADIENT_CLIP_NORM,
                    "actor_grad_norm_before_clip": actor_grad_norm,
                    "actor_grad_max_abs_before_clip": float(grad_stats_before_clip["grad_max_abs"]),
                    "actor_grad_mean_abs_before_clip": float(grad_stats_before_clip["grad_mean_abs"]),
                    "actor_grad_nonfinite_count_before_clip": int(grad_stats_before_clip["grad_nonfinite_count"]),
                    "actor_grad_is_finite_before_clip": bool(grad_stats_before_clip["grad_is_finite"]),
                    "actor_grad_would_clip": actor_grad_would_clip,
                    "actor_grad_clipped": actor_grad_clipped,
                    "actor_grad_norm_after_clip": float(grad_stats_after_clip["grad_norm"]),
                    "actor_grad_max_abs_after_clip": float(grad_stats_after_clip["grad_max_abs"]),
                    "actor_grad_mean_abs_after_clip": float(grad_stats_after_clip["grad_mean_abs"]),
                    "actor_grad_nonfinite_count_after_clip": int(grad_stats_after_clip["grad_nonfinite_count"]),
                    **component_gradient_stats,
                    **component_cosines,
                    **parameter_gradient_stats,
                    **rollout_diagnostics,
                }

                self._soft_update(self.q1, self.q1_target)
                self._soft_update(self.q2, self.q2_target)
                self._soft_update(self.actor, self.actor_target)

        return float(q1_current.mean().detach().cpu())

    def save_models(self, path: str) -> None:
        import os

        with self.update_lock:
            os.makedirs(path, exist_ok=True)
            torch.save(self.actor.state_dict(), f"{path}/actor.pt")
            torch.save(self.actor_target.state_dict(), f"{path}/actor_tgt.pt")
            torch.save(self.q1.state_dict(), f"{path}/critic1.pt")
            torch.save(self.q2.state_dict(), f"{path}/critic2.pt")
            torch.save(self.q1_target.state_dict(), f"{path}/critic1_tgt.pt")
            torch.save(self.q2_target.state_dict(), f"{path}/critic2_tgt.pt")
        print(f"[TD3VectorFieldAgent] Saved checkpoint to {path}")
