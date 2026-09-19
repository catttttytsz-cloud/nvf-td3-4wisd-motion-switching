"""ROS2 training node for the neural-vector-field TD3 command policy."""

from __future__ import annotations

import csv
import json
import os
import random
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import TwistStamped
from rosgraph_msgs.msg import Clock
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, UInt32

from neural_vector_field.command_sampler import sample_target_twist
from neural_vector_field.config import (
    ACTOR_DIAGNOSTIC_LOG_INTERVAL_UPDATES,
    CONSOLE_PRINT_GRADIENT_SUMMARY,
    CONSOLE_PRINT_INTERVAL_EPISODES,
    CONSOLE_PRINT_INTERVAL_STEPS,
    CONSOLE_PRINT_LOSS_COMPONENTS,
    CONSOLE_PRINT_Q_DIAGNOSTICS,
    CONSOLE_PRINT_REWARD_COMPONENTS,
    CONSOLE_PRINT_SVD_SUMMARY,
    DEFAULT_DATA_DIR,
    DOMAIN_RANDOMIZATION_BASE_SEED,
    DOMAIN_RANDOMIZATION_INTERVAL_EPISODES,
    DOMAIN_RANDOMIZATION_READY_TIMEOUT_SEC,
    DOMAIN_READY_TOPIC,
    DOMAIN_RESET_TOPIC,
    ENABLE_DOMAIN_RANDOMIZATION,
    GLOBAL_RANDOM_SEED,
    PYTHON_RANDOM_SEED,
    NUMPY_RANDOM_SEED,
    TORCH_RANDOM_SEED,
    DRIVING_JOINT_NAMES,
    EPISODE_SUMMARY_LOG_INTERVAL,
    ENABLE_ROLLING_CORRELATION_LOGGING,
    ROLLING_CORRELATION_WINDOW_EPISODES,
    K_EXP_ERROR,
    REWARD_ERROR_TAU,
    K_EXP_ERROR_FINE,
    REWARD_ERROR_TAU_FINE,
    K_LATERAL_SLIP,
    K_TANGENTIAL_ERROR,
    MAX_DRIVE_VEL,
    MAX_STEPS_PER_EPISODE,
    MODEL_SAVE_INTERVAL_EPISODES,
    NUM_EPISODES,
    SAVE_ACTOR_UPDATE_DIAGNOSTICS,
    SAVE_NUMERICAL_STABILITY_EVENTS,
    SAVE_STEP_REWARD_LOG,
    START_RANDOM_STEPS,
    LEARNING_UPDATE_INTERVAL_STEPS,
    CRITIC_UPDATES_PER_LEARNING_STEP,
    LEARNER_THREAD_ENABLED,
    LEARNER_REQUEST_QUEUE_MAXSIZE,
    LEARNER_RESULT_QUEUE_MAXSIZE,
    LEARNER_DROP_REQUEST_WHEN_FULL,
    LEARNER_JOIN_TIMEOUT_SEC,
    ASYNC_CSV_WRITER,
    CSV_QUEUE_MAXSIZE,
    CSV_FLUSH_INTERVAL_SEC,
    CSV_FLUSH_EVERY_N_ROWS,
    CSV_DROP_LOW_PRIORITY_WHEN_FULL,
    TIMING_PROFILE_ENABLED,
    TIMING_PROFILE_PRINT_INTERVAL_EPISODES,
    TRAINING_TRIGGER_MODE,
    CONTROL_STEP_SIM_TIME,
    SIM_TIME_TRIGGER_EPS,
    STABILITY_EVENT_ACTOR_GRAD_NORM_THRESHOLD,
    STABILITY_EVENT_CONDITION_PROXY_THRESHOLD,
    STABILITY_EVENT_SIGMA_MIN_THRESHOLD,
    STEER_LIMITS,
    STEERING_JOINT_NAMES,
    STEP_INTERVAL,
    WHEEL_POS,
    WHEEL_RADIUS,
    command_velocity_limits,
)
from neural_vector_field.four_wisd_kinematics import four_wisd_inverse_kinematics
from neural_vector_field.td3_vector_field_agent import TD3VectorFieldAgent
import neural_vector_field.config as config_module


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _min(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _pearson_corr(values_a, values_b) -> float:
    if len(values_a) < 3 or len(values_b) < 3:
        return 0.0
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    finite = np.isfinite(arr_a) & np.isfinite(arr_b)
    if int(finite.sum()) < 3:
        return 0.0
    arr_a = arr_a[finite]
    arr_b = arr_b[finite]
    if float(np.std(arr_a)) <= 1e-12 or float(np.std(arr_b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _append_csv_sync(path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


class AsyncCsvWriter:
    """Background CSV writer used to keep file I/O off the training loop."""

    def __init__(
        self,
        maxsize: int = 20000,
        flush_interval_sec: float = 1.0,
        flush_every_n_rows: int = 200,
        drop_low_priority_when_full: bool = True,
    ):
        self.queue: queue.Queue[tuple[str | None, dict[str, Any] | None, list[str] | None, str]] = queue.Queue(maxsize=maxsize)
        self.flush_interval_sec = max(float(flush_interval_sec), 0.05)
        self.flush_every_n_rows = max(int(flush_every_n_rows), 1)
        self.drop_low_priority_when_full = bool(drop_low_priority_when_full)
        self.stop_event = threading.Event()
        self.dropped_rows = 0
        self.written_rows = 0
        self._thread = threading.Thread(target=self._worker, name="async_csv_writer", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        path: str,
        row: dict[str, Any],
        fieldnames: list[str],
        priority: str = "normal",
    ) -> None:
        item = (path, dict(row), list(fieldnames), priority)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            if self.drop_low_priority_when_full and priority == "low":
                self.dropped_rows += 1
                return
            self.queue.put(item, timeout=0.25)

    def _worker(self) -> None:
        buffers: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        last_flush = time.perf_counter()
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                path, row, fields, _priority = self.queue.get(timeout=0.05)
            except queue.Empty:
                path = row = fields = None

            if path is not None and row is not None and fields is not None:
                key = (path, tuple(fields))
                buffers.setdefault(key, []).append(row)
                self.queue.task_done()

            now = time.perf_counter()
            should_flush = (now - last_flush) >= self.flush_interval_sec
            should_flush = should_flush or any(len(rows) >= self.flush_every_n_rows for rows in buffers.values())
            if should_flush:
                self._flush(buffers)
                last_flush = now

        self._flush(buffers)

    def _flush(self, buffers: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]]) -> None:
        for (path, fields_tuple), rows in list(buffers.items()):
            if not rows:
                continue
            fieldnames = list(fields_tuple)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            file_exists = os.path.exists(path)
            with open(path, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
            self.written_rows += len(rows)
            rows.clear()

    def close(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=5.0)


class VectorFieldTrainingNode(Node):
    def __init__(self):
        super().__init__("vector_field_training_node")
        self.apply_reproducibility_seed()

        self.num_episodes = NUM_EPISODES
        self.max_steps = MAX_STEPS_PER_EPISODE
        self.global_step = 0
        self.current_step = 0
        self.episode = 0
        self.episode_reward = 0.0
        self.step_interval = STEP_INTERVAL
        self.training_trigger_mode = TRAINING_TRIGGER_MODE
        self.control_step_sim_time = CONTROL_STEP_SIM_TIME
        self.current_sim_time: float | None = None
        self.last_step_sim_time: float | None = None
        self.wall_timer = None
        self.waiting_for_domain_ready = False
        self.domain_ready_request_wall_time: float | None = None
        self.domain_request_last_publish_wall_time: float | None = None
        self.domain_reset_request_count = 0
        self.last_domain_reset_seed = -1

        self.steering_joint_names = STEERING_JOINT_NAMES
        self.driving_joint_names = DRIVING_JOINT_NAMES

        self.agent = TD3VectorFieldAgent(
            command_low=np.full(8, -1.0, np.float32),
            command_high=np.full(8, 1.0, np.float32),
            env_dt=self.step_interval,
            vmax_norm=command_velocity_limits(),
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(DEFAULT_DATA_DIR, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_writer = (
            AsyncCsvWriter(
                maxsize=CSV_QUEUE_MAXSIZE,
                flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
                flush_every_n_rows=CSV_FLUSH_EVERY_N_ROWS,
                drop_low_priority_when_full=CSV_DROP_LOW_PRIORITY_WHEN_FULL,
            )
            if ASYNC_CSV_WRITER else None
        )

        self.async_learner_enabled = bool(LEARNER_THREAD_ENABLED)
        self.learning_request_queue: queue.Queue[dict[str, Any] | None] | None = None
        self.learning_result_queue: queue.Queue[dict[str, Any]] | None = None
        self.learning_stop_event = threading.Event()
        self.learner_thread: threading.Thread | None = None
        self.latest_q_value = 0.0
        rolling_window = max(int(ROLLING_CORRELATION_WINDOW_EPISODES), 3)
        self.rolling_q_values = deque(maxlen=rolling_window)
        self.rolling_episode_returns = deque(maxlen=rolling_window)
        self.rolling_negative_final_errors = deque(maxlen=rolling_window)
        self.total_learning_requests_enqueued = 0
        self.total_learning_requests_dropped = 0
        self.total_learning_updates_completed = 0
        self.total_learning_results_dropped = 0
        if self.async_learner_enabled:
            self.learning_request_queue = queue.Queue(maxsize=max(int(LEARNER_REQUEST_QUEUE_MAXSIZE), 1))
            self.learning_result_queue = queue.Queue(maxsize=max(int(LEARNER_RESULT_QUEUE_MAXSIZE), 1))
            self.learner_thread = threading.Thread(
                target=self.learner_worker,
                name="nvf_background_learner",
                daemon=True,
            )
            self.learner_thread.start()
            self.console(
                "Async learner enabled: "
                f"request_queue={LEARNER_REQUEST_QUEUE_MAXSIZE}, result_queue={LEARNER_RESULT_QUEUE_MAXSIZE}"
            )

        self.timing_profile_enabled = TIMING_PROFILE_ENABLED
        self.timing_totals: dict[str, float] = {}
        self.timing_counts: dict[str, int] = {}
        self.episode_wall_start = time.perf_counter()
        self.write_run_config()

        self.last_reward_info: dict[str, float] = {}
        # Store the latest actor-update diagnostics so periodic console output can
        # show loss/gradient/SVD/Q information even when the print step does not
        # coincide with a delayed TD3 actor update.
        self.last_actor_update_info: dict[str, Any] = {}
        self.reset_episode_accumulators()

        self.pub_driving = self.create_publisher(JointState, "/driving_joints_controller/command", 10)
        self.pub_steering = self.create_publisher(JointState, "/steering_joints_controller/command", 10)
        self.pub_result = self.create_publisher(Float32MultiArray, "result", 10)
        self.pub_get_action = self.create_publisher(Float32MultiArray, "get_action", 10)
        self.pub_domain_reset = self.create_publisher(UInt32, DOMAIN_RESET_TOPIC, 10)
        self.create_subscription(Bool, DOMAIN_READY_TOPIC, self.domain_ready_callback, 10)

        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.create_subscription(TwistStamped, "/agv/base_link_twist", self.twist_callback, 10)
        self.create_subscription(Float32MultiArray, "/agv/wheel_vector_data", self.wheel_vector_callback, 10)
        if self.training_trigger_mode == "sim_time_topic_gate":
            self.create_subscription(Clock, "/clock", self.clock_callback, 10)
            self.console(
                "Training trigger mode: sim_time_topic_gate "
                f"(control_step_sim_time={self.control_step_sim_time:.3f}s)"
            )
        elif self.training_trigger_mode == "wall_timer":
            self.wall_timer = self.create_timer(self.step_interval, self.try_step)
            self.console(
                f"Training trigger mode: wall_timer (step_interval={self.step_interval:.3f}s)"
            )
        else:
            raise ValueError(f"Unsupported TRAINING_TRIGGER_MODE: {self.training_trigger_mode}")

        self.target_twist = np.zeros(3, dtype=np.float32)
        self.base_velocity = np.zeros(3, dtype=np.float32)
        self.joint_state: JointState | None = None
        self.wheel_vector_data = None
        self.target_actuator_command = np.zeros(8, dtype=np.float32)

        self.new_base_velocity = np.zeros(3, dtype=np.float32)
        self.new_joint_state: JointState | None = None
        self.new_wheel_vector_data = None

        self.k_exp_error = K_EXP_ERROR
        self.reward_error_tau = REWARD_ERROR_TAU
        self.k_exp_error_fine = K_EXP_ERROR_FINE
        self.reward_error_tau_fine = REWARD_ERROR_TAU_FINE
        self.k_lateral_slip = K_LATERAL_SLIP
        self.k_tangential_error = K_TANGENTIAL_ERROR

        self.last_actor_update_info: dict[str, Any] = {}
        self.last_actor_diagnostic_info: dict[str, Any] = {}
        if ENABLE_DOMAIN_RANDOMIZATION:
            self.request_domain_reset(reason="initial")
        else:
            self.reset_episode()

    @staticmethod
    def console(message: str = "") -> None:
        """Print compact training diagnostics without ROS logger timestamps."""
        print(message, flush=True)



    @staticmethod
    def apply_reproducibility_seed() -> None:
        """Seed all stochastic components used by training and command sampling."""
        random.seed(int(PYTHON_RANDOM_SEED))
        np.random.seed(int(NUMPY_RANDOM_SEED))
        torch.manual_seed(int(TORCH_RANDOM_SEED))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(TORCH_RANDOM_SEED))
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    def make_domain_seed(self, episode: int) -> int:
        return int((int(DOMAIN_RANDOMIZATION_BASE_SEED) + int(episode)) % (2**32 - 1))

    def should_request_domain_reset(self) -> bool:
        if not ENABLE_DOMAIN_RANDOMIZATION:
            return False
        interval = int(DOMAIN_RANDOMIZATION_INTERVAL_EPISODES)
        if interval <= 0:
            return False
        return self.episode % interval == 0

    def request_domain_reset(self, reason: str = "scheduled") -> None:
        seed = self.make_domain_seed(self.episode)
        self.last_domain_reset_seed = int(seed)
        self.domain_reset_request_count += 1
        self.waiting_for_domain_ready = True
        self.domain_ready_request_wall_time = time.perf_counter()

        # Avoid storing a transition that crosses a real physics-domain reset.
        for attr_name in ("prev_observation", "prev_actor_command"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.last_step_sim_time = None
        self.new_joint_state = None
        self.new_base_velocity = np.zeros(3, dtype=np.float32)
        self.new_wheel_vector_data = None

        msg = UInt32()
        msg.data = int(seed)
        self.pub_domain_reset.publish(msg)
        self.domain_request_last_publish_wall_time = time.perf_counter()
        self.console(
            f"[domain reset request] reason={reason} episode={self.episode} seed={seed} "
            f"topic={DOMAIN_RESET_TOPIC}"
        )

    def domain_ready_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        if not self.waiting_for_domain_ready:
            return
        self.waiting_for_domain_ready = False
        wait_s = 0.0
        if self.domain_ready_request_wall_time is not None:
            wait_s = time.perf_counter() - self.domain_ready_request_wall_time
        self.domain_ready_request_wall_time = None
        self.domain_request_last_publish_wall_time = None
        self.console(
            f"[domain ready] episode={self.episode} seed={self.last_domain_reset_seed} wait={wait_s:.3f}s"
        )
        self.reset_episode()

    def maybe_timeout_domain_wait(self) -> None:
        if not self.waiting_for_domain_ready:
            return
        timeout = float(DOMAIN_RANDOMIZATION_READY_TIMEOUT_SEC)
        now = time.perf_counter()
        if self.domain_request_last_publish_wall_time is None or now - self.domain_request_last_publish_wall_time >= 1.0:
            msg = UInt32()
            msg.data = int(self.last_domain_reset_seed)
            self.pub_domain_reset.publish(msg)
            self.domain_request_last_publish_wall_time = now
        if timeout <= 0.0 or self.domain_ready_request_wall_time is None:
            return
        if now - self.domain_ready_request_wall_time < timeout:
            return
        self.console(
            f"[domain ready][WARN] timeout after {timeout:.1f}s; continuing with current Isaac domain. "
            f"seed={self.last_domain_reset_seed}"
        )
        self.waiting_for_domain_ready = False
        self.domain_ready_request_wall_time = None
        self.domain_request_last_publish_wall_time = None
        self.reset_episode()

    def write_run_config(self) -> None:
        """Save effective scalar/list configuration for reproducibility and reviewer response."""
        config_snapshot = {}
        for name in dir(config_module):
            if not name.isupper():
                continue
            value = getattr(config_module, name)
            if isinstance(value, np.ndarray):
                config_snapshot[name] = value.tolist()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                config_snapshot[name] = value
            elif isinstance(value, (tuple, list)):
                config_snapshot[name] = list(value)
        path = os.path.join(self.save_dir, "run_config.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config_snapshot, file, indent=2, sort_keys=True)
        self.console(f"Saved run config: {path}")

    def append_csv(
        self,
        path: str,
        row: dict[str, Any],
        fieldnames: list[str],
        priority: str = "normal",
    ) -> None:
        if self.csv_writer is not None:
            self.csv_writer.enqueue(path, row, fieldnames, priority=priority)
        else:
            _append_csv_sync(path, row, fieldnames)

    def enqueue_learning_updates(self, count: int) -> None:
        """Request background critic/actor updates without blocking the control step."""
        if not self.async_learner_enabled or self.learning_request_queue is None:
            return
        for _ in range(max(int(count), 0)):
            request = {
                "request_episode": int(self.episode),
                "request_step": int(self.current_step),
                "request_global_step": int(self.global_step),
            }
            try:
                self.learning_request_queue.put_nowait(request)
                self.episode_learning_requests_enqueued += 1
                self.total_learning_requests_enqueued += 1
            except queue.Full:
                self.episode_learning_requests_dropped += 1
                self.total_learning_requests_dropped += 1
                if not LEARNER_DROP_REQUEST_WHEN_FULL:
                    try:
                        self.learning_request_queue.put(request, timeout=0.001)
                        self.episode_learning_requests_enqueued += 1
                        self.total_learning_requests_enqueued += 1
                    except queue.Full:
                        pass

    def learner_worker(self) -> None:
        """Run TD3 updates in a background thread so control callbacks stay light."""
        assert self.learning_request_queue is not None
        assert self.learning_result_queue is not None
        while not self.learning_stop_event.is_set() or not self.learning_request_queue.empty():
            try:
                request = self.learning_request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if request is None:
                    continue
                # START_RANDOM_STEPS is checked before enqueueing, but the guard is
                # kept here so manual queue operations cannot trigger undersized samples.
                if self.agent.size() <= START_RANDOM_STEPS:
                    continue
                q_value = self.agent.update(episode=int(request.get("request_episode", 0)))
                update_info = dict(self.agent.last_update_info)
                update_info["async_learner"] = True
                update_info["async_q_value"] = float(q_value)
                update_info.update(request)
                try:
                    self.learning_result_queue.put_nowait(update_info)
                except queue.Full:
                    self.total_learning_results_dropped += 1
            except Exception as exc:  # keep the control loop alive and surface the error.
                error_info = {
                    "async_learner": True,
                    "learner_exception": repr(exc),
                    "actor_updated": False,
                    "q_value_mean": self.latest_q_value,
                }
                try:
                    self.learning_result_queue.put_nowait(error_info)
                except Exception:
                    pass
            finally:
                self.learning_request_queue.task_done()

    def drain_completed_learning_updates(self) -> None:
        """Move learner-thread results into episode statistics on the ROS thread."""
        if not self.async_learner_enabled or self.learning_result_queue is None:
            return
        while True:
            try:
                update_info = self.learning_result_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if "learner_exception" in update_info:
                    self.console(f"[learner][ERROR] {update_info['learner_exception']}")
                    continue
                self.latest_q_value = _safe_float(update_info.get("q_value_mean"), self.latest_q_value)
                self.record_update_statistics(update_info)
                self.episode_learning_updates_completed += 1
                self.total_learning_updates_completed += 1
            finally:
                self.learning_result_queue.task_done()

    def close_learner(self) -> None:
        if not self.async_learner_enabled:
            return
        self.learning_stop_event.set()
        if self.learner_thread is not None:
            self.learner_thread.join(timeout=max(float(LEARNER_JOIN_TIMEOUT_SEC), 0.1))
            self.learner_thread = None
        self.drain_completed_learning_updates()

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        if not self.timing_profile_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timing_totals[name] = self.timing_totals.get(name, 0.0) + elapsed
            self.timing_counts[name] = self.timing_counts.get(name, 0) + 1

    def reset_timing_profile(self) -> None:
        self.timing_totals = {}
        self.timing_counts = {}
        self.episode_wall_start = time.perf_counter()

    def timing_summary_string(self) -> str:
        if not self.timing_totals:
            return ""
        ordered = sorted(self.timing_totals.items(), key=lambda item: item[1], reverse=True)
        parts = []
        for key, total in ordered[:8]:
            count = max(self.timing_counts.get(key, 1), 1)
            parts.append(f"{key}={total:.3f}s/{count}")
        if self.csv_writer is not None:
            parts.append(f"csv_written={self.csv_writer.written_rows}")
            parts.append(f"csv_dropped={self.csv_writer.dropped_rows}")
        return " | ".join(parts)

    def reset_episode_accumulators(self) -> None:
        self.episode_rewards: list[float] = []
        self.episode_actuator_errors: list[float] = []
        self.episode_lateral_slip: list[float] = []
        self.episode_tangential_error: list[float] = []
        self.episode_target_state_reward: list[float] = []
        self.episode_target_state_reward_main: list[float] = []
        self.episode_target_state_reward_fine: list[float] = []
        self.episode_q_values: list[float] = []
        self.episode_critic_losses: list[float] = []
        self.episode_actor_update_infos: list[dict[str, Any]] = []
        self.episode_nonfinite_events = 0
        self.episode_would_clip_count = 0
        self.episode_sim_dt_values: list[float] = []
        self.episode_skipped_intervals = 0
        self.episode_learning_requests_enqueued = 0
        self.episode_learning_requests_dropped = 0
        self.episode_learning_updates_completed = 0

    def joint_state_callback(self, msg: JointState) -> None:
        self.new_joint_state = msg
        self.maybe_try_step()

    def twist_callback(self, msg: TwistStamped) -> None:
        self.new_base_velocity = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z],
            dtype=np.float32,
        )
        self.maybe_try_step()

    def wheel_vector_callback(self, msg: Float32MultiArray) -> None:
        # Topic gating synchronizes the control step with the required ROS inputs.
        # The current training reward no longer uses the message contents.
        self.new_wheel_vector_data = msg.data
        self.maybe_try_step()

    def clock_callback(self, msg: Clock) -> None:
        self.current_sim_time = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9
        self.maybe_try_step()

    def topics_ready(self) -> bool:
        return (
            self.new_joint_state is not None
            and self.new_base_velocity is not None
            and self.new_wheel_vector_data is not None
        )

    def commit_latest_topic_data(self) -> None:
        self.joint_state = self.new_joint_state
        self.base_velocity = self.new_base_velocity
        self.wheel_vector_data = self.new_wheel_vector_data

    def maybe_try_step(self) -> None:
        if self.training_trigger_mode != "sim_time_topic_gate":
            return
        self.maybe_timeout_domain_wait()
        if self.waiting_for_domain_ready:
            return
        if self.current_sim_time is None or not self.topics_ready():
            return
        if self.last_step_sim_time is None:
            # Allow the first available synchronized topic set to trigger a step.
            self.last_step_sim_time = self.current_sim_time - self.control_step_sim_time
        sim_dt_actual = self.current_sim_time - self.last_step_sim_time
        if sim_dt_actual < self.control_step_sim_time - SIM_TIME_TRIGGER_EPS:
            return
        skipped = max(0, int(sim_dt_actual / max(self.control_step_sim_time, 1e-9)) - 1)
        self.episode_sim_dt_values.append(float(sim_dt_actual))
        self.episode_skipped_intervals += int(skipped)
        self.commit_latest_topic_data()
        self.last_step_sim_time = self.current_sim_time
        self.step_once()

    def try_step(self) -> None:
        if self.training_trigger_mode != "wall_timer":
            return
        self.maybe_timeout_domain_wait()
        if self.waiting_for_domain_ready:
            return
        if self.topics_ready():
            self.commit_latest_topic_data()
            self.step_once()

    def step_once(self) -> None:
        # Pull completed learner-thread updates before starting the control step.
        # This keeps CSV/console statistics current without waiting for training.
        self.drain_completed_learning_updates()
        with self.profile("observation"):
            observation = self.get_observation()
        reward_prev = 0.0
        done_prev = False
        q_value = self.latest_q_value

        with self.profile("reward_buffer"):
            if hasattr(self, "prev_observation"):
                reward_prev = self.compute_reward(observation)
                self.agent.push(
                    self.prev_observation,
                    self.prev_actor_command,
                    reward_prev,
                    observation,
                    done_prev,
                )

        with self.profile("select_command"):
            actor_command = self.agent.select_command(observation, self.episode)
        with self.profile("publish_command"):
            self.publish_actor_command(actor_command)

        self.prev_observation = observation
        self.prev_actor_command = actor_command

        self.global_step += 1
        self.current_step += 1
        self.episode_reward += reward_prev
        self.record_step_statistics(reward_prev)

        update_info = {"actor_updated": False}
        should_learn = (
            self.agent.size() > START_RANDOM_STEPS
            and LEARNING_UPDATE_INTERVAL_STEPS > 0
            and self.global_step % LEARNING_UPDATE_INTERVAL_STEPS == 0
        )
        if should_learn:
            if self.async_learner_enabled:
                with self.profile("learning_enqueue"):
                    self.enqueue_learning_updates(max(int(CRITIC_UPDATES_PER_LEARNING_STEP), 1))
            else:
                with self.profile("learning_update"):
                    for _ in range(max(int(CRITIC_UPDATES_PER_LEARNING_STEP), 1)):
                        q_value = self.agent.update(episode=int(self.episode))
                        self.latest_q_value = q_value
                        update_info = dict(self.agent.last_update_info)
                        self.record_update_statistics(update_info)

        # Drain again so a fast learner update can be reflected in this step's print/log.
        self.drain_completed_learning_updates()
        q_value = self.latest_q_value

        with self.profile("publish_debug_action"):
            action_msg = Float32MultiArray()
            action_msg.data = np.concatenate(
                [self.target_twist, self.base_velocity, actor_command, [reward_prev]]
            ).astype(np.float32).tolist()
            self.pub_get_action.publish(action_msg)

        if SAVE_STEP_REWARD_LOG:
            with self.profile("step_csv_enqueue"):
                self.write_step_reward_log(reward_prev, q_value, actor_command, update_info)

        with self.profile("console_print"):
            self.maybe_print_console(reward_prev, q_value, update_info)

        if done_prev or self.current_step >= self.max_steps:
            with self.profile("episode_end"):
                result_msg = Float32MultiArray()
                mean_critic_loss = _mean(self.episode_critic_losses)
                result_msg.data = [float(self.episode_reward), float(mean_critic_loss)]
                self.pub_result.publish(result_msg)
                self.write_episode_summary(q_value)
                wall_time = max(time.perf_counter() - self.episode_wall_start, 1e-9)
                effective_rtf = (self.current_step * self.control_step_sim_time) / wall_time
                self.console(
                    f"[Ep {self.episode} done] R_total={self.episode_reward:.2f} "
                    f"steps={self.current_step} wall={wall_time:.2f}s train_rtf={effective_rtf:.2f}"
                )
                if (
                    self.timing_profile_enabled
                    and TIMING_PROFILE_PRINT_INTERVAL_EPISODES > 0
                    and self.episode % TIMING_PROFILE_PRINT_INTERVAL_EPISODES == 0
                ):
                    self.console("  timing: " + self.timing_summary_string())

                self.episode += 1
                if self.episode % MODEL_SAVE_INTERVAL_EPISODES == 0:
                    self.save_checkpoint()
            if self.should_request_domain_reset():
                self.request_domain_reset(reason="scheduled")
            else:
                self.reset_episode()

    def record_step_statistics(self, reward: float) -> None:
        self.episode_rewards.append(float(reward))
        info = self.last_reward_info
        self.episode_actuator_errors.append(_safe_float(info.get("reward_actuator_error_norm")))
        self.episode_lateral_slip.append(_safe_float(info.get("reward_lateral_slip_penalty")))
        self.episode_tangential_error.append(_safe_float(info.get("reward_tangential_velocity_penalty")))
        self.episode_target_state_reward.append(_safe_float(info.get("reward_target_state")))
        self.episode_target_state_reward_main.append(_safe_float(info.get("reward_target_state_main")))
        self.episode_target_state_reward_fine.append(_safe_float(info.get("reward_target_state_fine")))

    def record_update_statistics(self, update_info: dict[str, Any]) -> None:
        self.episode_q_values.append(_safe_float(update_info.get("q_value_mean")))
        self.episode_critic_losses.append(_safe_float(update_info.get("critic_loss")))
        if not update_info.get("actor_updated", False):
            return
        # Keep a copy for console reporting at fixed step intervals. TD3 updates
        # the actor only every POLICY_UPDATE_FREQ critic updates, so the console
        # print step may otherwise see a critic-only update_info and omit all
        # actor diagnostics.
        self.last_actor_update_info = dict(update_info)
        if update_info.get("grad_q_term_grad_norm") is not None:
            self.last_actor_diagnostic_info = dict(update_info)
        self.episode_actor_update_infos.append(update_info)
        if bool(update_info.get("actor_grad_would_clip", False)):
            self.episode_would_clip_count += 1
        nonfinite = int(update_info.get("actor_grad_nonfinite_count_before_clip", 0) or 0)
        nonfinite += int(update_info.get("cms_singular_value_nonfinite_count", 0) or 0)
        nonfinite += int(update_info.get("cms_matrix_nonfinite_count", 0) or 0)
        if nonfinite > 0:
            self.episode_nonfinite_events += 1

        actor_update_index = int(update_info.get("actor_update_index", 0) or 0)
        if SAVE_ACTOR_UPDATE_DIAGNOSTICS and actor_update_index % ACTOR_DIAGNOSTIC_LOG_INTERVAL_UPDATES == 0:
            self.write_actor_update_diagnostics(update_info)
        if SAVE_NUMERICAL_STABILITY_EVENTS:
            self.maybe_write_stability_event(update_info)

    def maybe_print_console(self, reward: float, q_value: float, update_info: dict[str, Any]) -> None:
        if CONSOLE_PRINT_INTERVAL_EPISODES <= 0 or CONSOLE_PRINT_INTERVAL_STEPS <= 0:
            return
        if self.episode % CONSOLE_PRINT_INTERVAL_EPISODES != 0:
            return
        if self.current_step % CONSOLE_PRINT_INTERVAL_STEPS != 0:
            return

        # TD3 updates the actor only every POLICY_UPDATE_FREQ critic updates.
        # Use the latest actor diagnostics so periodic prints are informative
        # even if this exact step is critic-only.
        actor_info = update_info if update_info.get("actor_updated", False) else self.last_actor_update_info
        has_actor_info = bool(actor_info.get("actor_updated", False))

        lines = [
            f"[Ep {self.episode} | step {self.current_step}/{self.max_steps}] "
            f"R={reward:.3f} | Q={q_value:.3f}",
        ]

        if has_actor_info:
            source = "current" if update_info.get("actor_updated", False) else "last"
            lines.append(
                f"  actor_update: idx={int(actor_info.get('actor_update_index', 0) or 0)} "
                f"source={source}"
            )

        if CONSOLE_PRINT_REWARD_COMPONENTS:
            lines.append(
                "  reward: "
                f"err={_safe_float(self.last_reward_info.get('reward_actuator_error_norm')):.3e} | "
                f"rE={_safe_float(self.last_reward_info.get('reward_target_state')):.3e} "
                f"(main={_safe_float(self.last_reward_info.get('reward_target_state_main')):.3e}, "
                f"fine={_safe_float(self.last_reward_info.get('reward_target_state_fine')):.3e}) | "
                f"lat={_safe_float(self.last_reward_info.get('reward_lateral_slip_penalty')):.3e} | "
                f"tan={_safe_float(self.last_reward_info.get('reward_tangential_velocity_penalty')):.3e}"
            )

        if has_actor_info and CONSOLE_PRINT_LOSS_COMPONENTS:
            lines.append(
                "  loss: "
                f"total={_safe_float(actor_info.get('actor_loss')):.3e} | "
                f"-Q={_safe_float(actor_info.get('q_term')):.3e} | "
                f"SA={_safe_float(actor_info.get('state_attractor_term')):.3e} | "
                f"CMS={_safe_float(actor_info.get('control_misalignment_term')):.3e}"
            )

        if has_actor_info and CONSOLE_PRINT_GRADIENT_SUMMARY:
            grad_info = actor_info
            grad_source = "current" if update_info.get("actor_updated", False) else "last"
            if grad_info.get("grad_q_term_grad_norm") is None and self.last_actor_diagnostic_info:
                grad_info = self.last_actor_diagnostic_info
                grad_source = "last_diag"
            lines.append(
                "  grad: "
                f"source={grad_source} | "
                f"total={_safe_float(grad_info.get('actor_grad_norm_before_clip')):.3e} | "
                f"Q={_safe_float(grad_info.get('grad_q_term_grad_norm')):.3e} | "
                f"SA={_safe_float(grad_info.get('grad_state_attractor_term_grad_norm')):.3e} | "
                f"CMS={_safe_float(grad_info.get('grad_control_misalignment_term_grad_norm')):.3e} | "
                f"would_clip={int(bool(grad_info.get('actor_grad_would_clip', False)))}"
            )

        if has_actor_info and CONSOLE_PRINT_SVD_SUMMARY:
            lines.append(
                "  svd:  "
                "sigma_min(mean/min)="
                f"{_safe_float(actor_info.get('cms_sigma_min_mean')):.3e}/"
                f"{_safe_float(actor_info.get('cms_sigma_min_min')):.3e} | "
                f"cond_p95={_safe_float(actor_info.get('cms_condition_proxy_p95_mean')):.3e} | "
                "nonfinite(A/S/g)="
                f"{int(actor_info.get('cms_matrix_nonfinite_count', 0) or 0)}/"
                f"{int(actor_info.get('cms_singular_value_nonfinite_count', 0) or 0)}/"
                f"{int(actor_info.get('actor_grad_nonfinite_count_before_clip', 0) or 0)}"
            )

        if has_actor_info and CONSOLE_PRINT_Q_DIAGNOSTICS:
            lines.append(
                "  qdiag:"
                f" Qgap={_safe_float(actor_info.get('q_pi_gap_mean')):.3e} | "
                f"cos(Q,SA)={_safe_float(actor_info.get('cos_q_state_attractor')):.2f} | "
                f"cos(Q,CMS)={_safe_float(actor_info.get('cos_q_control_misalignment')):.2f}"
            )

        if not has_actor_info and (
            CONSOLE_PRINT_LOSS_COMPONENTS
            or CONSOLE_PRINT_GRADIENT_SUMMARY
            or CONSOLE_PRINT_SVD_SUMMARY
            or CONSOLE_PRINT_Q_DIAGNOSTICS
        ):
            lines.append("  actor_update: unavailable")

        self.console("\n".join(lines))

    def get_observation(self) -> np.ndarray:
        if self.joint_state is None:
            steering_raw = np.zeros(4)
            driving_raw = np.zeros(4)
        else:
            position_dict = dict(zip(self.joint_state.name, self.joint_state.position))
            velocity_dict = dict(zip(self.joint_state.name, self.joint_state.velocity))
            steering_raw = np.array([position_dict.get(name, 0.0) for name in self.steering_joint_names])
            driving_raw = np.array([velocity_dict.get(name, 0.0) for name in self.driving_joint_names])

        steering_norm = np.array([
            2.0 * (value - low) / (high - low) - 1.0
            for value, (low, high) in zip(steering_raw, STEER_LIMITS)
        ])
        driving_norm = np.clip(driving_raw / MAX_DRIVE_VEL, -1.0, 1.0)

        steering_error = self.target_actuator_command[:4] - steering_raw
        driving_error = self.target_actuator_command[4:] - driving_raw
        steering_error_norm = np.clip(
            steering_error / ((STEER_LIMITS[:, 1] - STEER_LIMITS[:, 0]) / 2.0),
            -1.0,
            1.0,
        )
        driving_error_norm = np.clip(driving_error / MAX_DRIVE_VEL, -1.0, 1.0)

        return np.concatenate(
            [steering_norm, driving_norm, steering_error_norm, driving_error_norm]
        ).astype(np.float32)

    def publish_actor_command(self, actor_command: np.ndarray) -> None:
        steering_command = np.array([
            0.5 * (value + 1.0) * (high - low) + low
            for value, (low, high) in zip(actor_command[:4], STEER_LIMITS)
        ])
        driving_command = np.clip(actor_command[4:] * MAX_DRIVE_VEL, -MAX_DRIVE_VEL, MAX_DRIVE_VEL)

        steering_msg = JointState()
        steering_msg.name = self.steering_joint_names
        steering_msg.position = steering_command.tolist()

        driving_msg = JointState()
        driving_msg.name = self.driving_joint_names
        driving_msg.velocity = driving_command.tolist()

        self.pub_steering.publish(steering_msg)
        self.pub_driving.publish(driving_msg)

    def compute_reward(self, state: np.ndarray) -> float:
        if self.joint_state is None:
            steering_angle = np.zeros(4)
            wheel_velocity = np.zeros(4)
        else:
            position_dict = dict(zip(self.joint_state.name, self.joint_state.position))
            velocity_dict = dict(zip(self.joint_state.name, self.joint_state.velocity))
            steering_angle = np.array([position_dict.get(name, 0.0) for name in self.steering_joint_names])
            wheel_velocity = np.array([velocity_dict.get(name, 0.0) for name in self.driving_joint_names])

        vx, vy, wz = self.base_velocity
        x_pos = WHEEL_POS[:, 0]
        y_pos = WHEEL_POS[:, 1]
        wheel_center_vx = vx - wz * y_pos
        wheel_center_vy = vy + wz * x_pos

        cos_d = np.cos(steering_angle)
        sin_d = np.sin(steering_angle)
        tangent_x = cos_d
        tangent_y = sin_d
        normal_x = -sin_d
        normal_y = cos_d

        lateral_velocity = normal_x * wheel_center_vx + normal_y * wheel_center_vy
        lateral_slip_penalty = np.abs(lateral_velocity).sum()

        wheel_linear_velocity = wheel_velocity * WHEEL_RADIUS
        body_tangent_velocity = tangent_x * wheel_center_vx + tangent_y * wheel_center_vy
        tangential_velocity_penalty = np.abs(wheel_linear_velocity - body_tangent_velocity).sum()

        actuator_error_norm = np.linalg.norm(state[-8:])
        target_state_reward_main = self.k_exp_error * np.exp(
            -actuator_error_norm / max(self.reward_error_tau, 1.0e-9)
        )
        target_state_reward_fine = self.k_exp_error_fine * np.exp(
            -actuator_error_norm / max(self.reward_error_tau_fine, 1.0e-9)
        )
        target_state_reward = target_state_reward_main + target_state_reward_fine
        reward = target_state_reward - (
            self.k_lateral_slip * lateral_slip_penalty
            + self.k_tangential_error * tangential_velocity_penalty
        )
        self.last_reward_info = {
            "reward_actuator_error_norm": float(actuator_error_norm),
            "reward_target_state": float(target_state_reward),
            "reward_target_state_main": float(target_state_reward_main),
            "reward_target_state_fine": float(target_state_reward_fine),
            "reward_error_tau": float(self.reward_error_tau),
            "reward_error_tau_fine": float(self.reward_error_tau_fine),
            "reward_lateral_slip_penalty": float(lateral_slip_penalty),
            "reward_tangential_velocity_penalty": float(tangential_velocity_penalty),
            "reward_lateral_slip_weighted": float(self.k_lateral_slip * lateral_slip_penalty),
            "reward_tangential_velocity_weighted": float(self.k_tangential_error * tangential_velocity_penalty),
        }
        return float(reward)

    def write_step_reward_log(
        self,
        reward: float,
        q_value: float,
        actor_command: np.ndarray,
        update_info: dict[str, Any],
    ) -> None:
        row = {
            "episode": self.episode,
            "step": self.current_step,
            "global_step": self.global_step,
            "reward": float(reward),
            "episode_reward": float(self.episode_reward),
            "q_value": float(q_value),
            "target_twist_x": float(self.target_twist[0]),
            "target_twist_y": float(self.target_twist[1]),
            "target_twist_wz": float(self.target_twist[2]),
            "global_random_seed": int(GLOBAL_RANDOM_SEED),
            "python_random_seed": int(PYTHON_RANDOM_SEED),
            "numpy_random_seed": int(NUMPY_RANDOM_SEED),
            "torch_random_seed": int(TORCH_RANDOM_SEED),
            "last_domain_reset_seed": int(self.last_domain_reset_seed),
            "domain_reset_request_count": int(self.domain_reset_request_count),
            "base_velocity_x": float(self.base_velocity[0]),
            "base_velocity_y": float(self.base_velocity[1]),
            "base_velocity_wz": float(self.base_velocity[2]),
            "actor_command_norm": float(np.linalg.norm(actor_command)),
            "observation_error_norm": float(np.linalg.norm(self.prev_observation[-8:])),
            **self.last_reward_info,
            "actor_updated": bool(update_info.get("actor_updated", False)),
        }
        fieldnames = list(row.keys())
        self.append_csv(os.path.join(self.save_dir, "step_reward_log.csv"), row, fieldnames, priority="low")

    def write_actor_update_diagnostics(self, update_info: dict[str, Any]) -> None:
        fields = [
            "episode", "step", "global_step", "actor_update_index", "total_update_step",
            "actor_loss", "q_term", "state_attractor_term", "control_misalignment_term",
            "state_attractor_loss", "control_misalignment_loss",
            "grad_actor_loss_grad_norm", "grad_q_term_grad_norm",
            "grad_state_attractor_term_grad_norm", "grad_control_misalignment_term_grad_norm",
            "actor_grad_norm_before_clip", "actor_grad_would_clip", "actor_grad_nonfinite_count_before_clip",
            "cos_q_state_attractor", "cos_q_control_misalignment", "cos_state_attractor_control_misalignment",
            "cos_q_actor_loss", "cos_state_attractor_actor_loss", "cos_control_misalignment_actor_loss",
            "q1_pi_mean", "q2_pi_mean", "q_pi_gap_mean", "q1_current_mean", "q2_current_mean",
            "q_current_gap_mean", "critic_loss", "td_error1_abs_mean", "td_error2_abs_mean",
            "cms_sigma_min_mean", "cms_sigma_min_min", "cms_sigma_min_p05_mean",
            "cms_sigma_min_p50_mean", "cms_sigma_min_p95_mean", "cms_condition_proxy_p95_mean",
            "cms_condition_proxy_max", "cms_singular_value_nonfinite_count", "cms_matrix_nonfinite_count",
        ]
        row = {"episode": self.episode, "step": self.current_step, "global_step": self.global_step}
        row.update(update_info)
        self.append_csv(os.path.join(self.save_dir, "actor_update_diagnostics.csv"), row, fields, priority="normal")

    def maybe_write_stability_event(self, update_info: dict[str, Any]) -> None:
        event_reasons = []
        if int(update_info.get("actor_grad_nonfinite_count_before_clip", 0) or 0) > 0:
            event_reasons.append("actor_grad_nonfinite")
        if int(update_info.get("cms_singular_value_nonfinite_count", 0) or 0) > 0:
            event_reasons.append("svd_nonfinite")
        if int(update_info.get("cms_matrix_nonfinite_count", 0) or 0) > 0:
            event_reasons.append("matrix_nonfinite")
        if _safe_float(update_info.get("actor_grad_norm_before_clip")) > STABILITY_EVENT_ACTOR_GRAD_NORM_THRESHOLD:
            event_reasons.append("actor_grad_spike")
        if _safe_float(update_info.get("cms_condition_proxy_max")) > STABILITY_EVENT_CONDITION_PROXY_THRESHOLD:
            event_reasons.append("condition_proxy_spike")
        if 0.0 < _safe_float(update_info.get("cms_sigma_min_min"), default=1.0) < STABILITY_EVENT_SIGMA_MIN_THRESHOLD:
            event_reasons.append("sigma_min_extremely_small")
        if not event_reasons:
            return

        fields = [
            "episode", "step", "global_step", "actor_update_index", "event_reason",
            "actor_grad_norm_before_clip", "grad_q_term_grad_norm", "grad_state_attractor_term_grad_norm",
            "grad_control_misalignment_term_grad_norm", "cms_sigma_min_min", "cms_condition_proxy_max",
            "cms_singular_value_nonfinite_count", "cms_matrix_nonfinite_count",
            "actor_grad_nonfinite_count_before_clip",
        ]
        row = {
            "episode": self.episode,
            "step": self.current_step,
            "global_step": self.global_step,
            "event_reason": ";".join(event_reasons),
        }
        row.update(update_info)
        self.append_csv(os.path.join(self.save_dir, "numerical_stability_events.csv"), row, fields, priority="high")

    def write_episode_summary(self, q_value: float) -> None:
        self.drain_completed_learning_updates()
        q_value = self.latest_q_value
        if EPISODE_SUMMARY_LOG_INTERVAL <= 0 or self.episode % EPISODE_SUMMARY_LOG_INTERVAL != 0:
            return
        updates = self.episode_actor_update_infos
        def mean_update(key: str) -> float:
            return _mean([_safe_float(info.get(key)) for info in updates])
        def max_update(key: str) -> float:
            return _max([_safe_float(info.get(key)) for info in updates])
        def min_update(key: str) -> float:
            vals = [_safe_float(info.get(key)) for info in updates]
            return _min(vals)

        current_final_error = self.episode_actuator_errors[-1] if self.episode_actuator_errors else 0.0
        if ENABLE_ROLLING_CORRELATION_LOGGING:
            self.rolling_q_values.append(float(q_value))
            self.rolling_episode_returns.append(float(self.episode_reward))
            self.rolling_negative_final_errors.append(float(-current_final_error))
        rolling_q_return_corr = _pearson_corr(self.rolling_q_values, self.rolling_episode_returns)
        rolling_q_neg_final_error_corr = _pearson_corr(self.rolling_q_values, self.rolling_negative_final_errors)
        rolling_corr_count = len(self.rolling_q_values)

        fields = [
            "episode", "steps", "episode_return", "mean_reward", "final_reward", "q_value_last",
            "rolling_q_return_corr", "rolling_q_neg_final_error_corr", "rolling_corr_count",
            "training_trigger_mode", "current_sim_time", "episode_wall_time_sec", "effective_train_rtf",
            "sim_dt_mean", "sim_dt_max", "skipped_control_intervals",
            "learner_thread_enabled", "learner_requests_enqueued", "learner_requests_dropped",
            "learner_updates_completed", "learner_queue_backlog", "learner_total_requests_dropped",
            "mean_actuator_error", "final_actuator_error", "max_actuator_error",
            "mean_lateral_slip", "mean_tangential_error", "mean_target_state_reward",
            "mean_target_state_reward_main", "mean_target_state_reward_fine",
            "actor_updates", "would_clip_count", "nonfinite_event_count",
            "actor_loss_mean", "q_term_mean", "state_attractor_term_mean", "control_misalignment_term_mean",
            "grad_actor_mean", "grad_q_mean", "grad_sa_mean", "grad_cms_mean",
            "cos_q_sa_mean", "cos_q_cms_mean", "cos_sa_cms_mean",
            "q1_pi_mean", "q2_pi_mean", "q_pi_gap_mean", "critic_loss_mean", "td_error1_abs_mean",
            "sigma_min_mean", "sigma_min_min", "condition_proxy_p95_mean", "condition_proxy_max",
            "svd_nonfinite_count", "matrix_nonfinite_count", "grad_nonfinite_count",
            "target_twist_x", "target_twist_y", "target_twist_wz",
            "global_random_seed", "python_random_seed", "numpy_random_seed", "torch_random_seed",
            "last_domain_reset_seed", "domain_reset_request_count",
        ]
        row = {
            "episode": self.episode,
            "steps": self.current_step,
            "episode_return": float(self.episode_reward),
            "mean_reward": _mean(self.episode_rewards),
            "final_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "q_value_last": float(q_value),
            "rolling_q_return_corr": rolling_q_return_corr,
            "rolling_q_neg_final_error_corr": rolling_q_neg_final_error_corr,
            "rolling_corr_count": int(rolling_corr_count),
            "training_trigger_mode": self.training_trigger_mode,
            "current_sim_time": _safe_float(self.current_sim_time),
            "episode_wall_time_sec": max(time.perf_counter() - self.episode_wall_start, 0.0),
            "effective_train_rtf": (self.current_step * self.control_step_sim_time) / max(time.perf_counter() - self.episode_wall_start, 1e-9),
            "sim_dt_mean": _mean(self.episode_sim_dt_values),
            "sim_dt_max": _max(self.episode_sim_dt_values),
            "skipped_control_intervals": int(self.episode_skipped_intervals),
            "learner_thread_enabled": bool(self.async_learner_enabled),
            "learner_requests_enqueued": int(self.episode_learning_requests_enqueued),
            "learner_requests_dropped": int(self.episode_learning_requests_dropped),
            "learner_updates_completed": int(self.episode_learning_updates_completed),
            "learner_queue_backlog": int(self.learning_request_queue.qsize()) if self.learning_request_queue is not None else 0,
            "learner_total_requests_dropped": int(self.total_learning_requests_dropped),
            "mean_actuator_error": _mean(self.episode_actuator_errors),
            "final_actuator_error": current_final_error,
            "max_actuator_error": _max(self.episode_actuator_errors),
            "mean_lateral_slip": _mean(self.episode_lateral_slip),
            "mean_tangential_error": _mean(self.episode_tangential_error),
            "mean_target_state_reward": _mean(self.episode_target_state_reward),
            "mean_target_state_reward_main": _mean(self.episode_target_state_reward_main),
            "mean_target_state_reward_fine": _mean(self.episode_target_state_reward_fine),
            "actor_updates": len(updates),
            "would_clip_count": self.episode_would_clip_count,
            "nonfinite_event_count": self.episode_nonfinite_events,
            "actor_loss_mean": mean_update("actor_loss"),
            "q_term_mean": mean_update("q_term"),
            "state_attractor_term_mean": mean_update("state_attractor_term"),
            "control_misalignment_term_mean": mean_update("control_misalignment_term"),
            "grad_actor_mean": mean_update("actor_grad_norm_before_clip"),
            "grad_q_mean": mean_update("grad_q_term_grad_norm"),
            "grad_sa_mean": mean_update("grad_state_attractor_term_grad_norm"),
            "grad_cms_mean": mean_update("grad_control_misalignment_term_grad_norm"),
            "cos_q_sa_mean": mean_update("cos_q_state_attractor"),
            "cos_q_cms_mean": mean_update("cos_q_control_misalignment"),
            "cos_sa_cms_mean": mean_update("cos_state_attractor_control_misalignment"),
            "q1_pi_mean": mean_update("q1_pi_mean"),
            "q2_pi_mean": mean_update("q2_pi_mean"),
            "q_pi_gap_mean": mean_update("q_pi_gap_mean"),
            "critic_loss_mean": mean_update("critic_loss"),
            "td_error1_abs_mean": mean_update("td_error1_abs_mean"),
            "sigma_min_mean": mean_update("cms_sigma_min_mean"),
            "sigma_min_min": min_update("cms_sigma_min_min"),
            "condition_proxy_p95_mean": mean_update("cms_condition_proxy_p95_mean"),
            "condition_proxy_max": max_update("cms_condition_proxy_max"),
            "svd_nonfinite_count": int(sum(int(info.get("cms_singular_value_nonfinite_count", 0) or 0) for info in updates)),
            "matrix_nonfinite_count": int(sum(int(info.get("cms_matrix_nonfinite_count", 0) or 0) for info in updates)),
            "grad_nonfinite_count": int(sum(int(info.get("actor_grad_nonfinite_count_before_clip", 0) or 0) for info in updates)),
            "target_twist_x": float(self.target_twist[0]),
            "target_twist_y": float(self.target_twist[1]),
            "target_twist_wz": float(self.target_twist[2]),
            "global_random_seed": int(GLOBAL_RANDOM_SEED),
            "python_random_seed": int(PYTHON_RANDOM_SEED),
            "numpy_random_seed": int(NUMPY_RANDOM_SEED),
            "torch_random_seed": int(TORCH_RANDOM_SEED),
            "last_domain_reset_seed": int(self.last_domain_reset_seed),
            "domain_reset_request_count": int(self.domain_reset_request_count),
        }
        self.append_csv(os.path.join(self.save_dir, "episode_summary.csv"), row, fields, priority="high")

    def save_checkpoint(self) -> None:
        model_dir = os.path.join(self.save_dir, f"epi_{self.episode}")
        self.agent.save_models(path=model_dir)
        self.console(f"Saved checkpoint at episode {self.episode}")

    def reset_episode(self) -> None:
        self.reset_timing_profile()
        self.current_step = 0
        self.episode_reward = 0.0
        self.reset_episode_accumulators()
        self.target_twist = np.array(sample_target_twist(), dtype=np.float32)
        self.target_actuator_command = four_wisd_inverse_kinematics(*self.target_twist).astype(np.float32)
        self.console(
            "\n" + "=" * 72 +
            f"\nEP {self.episode} reset | target_twist = "
            f"[{self.target_twist[0]: .3e}, {self.target_twist[1]: .3e}, {self.target_twist[2]: .3e}]"
        )

    def destroy_node(self) -> bool:
        self.close_learner()
        if getattr(self, "csv_writer", None) is not None:
            self.csv_writer.close()
            self.csv_writer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VectorFieldTrainingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
