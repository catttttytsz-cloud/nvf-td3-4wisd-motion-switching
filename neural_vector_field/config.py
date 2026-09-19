"""Central configuration for neural-vector-field training and evaluation."""

from __future__ import annotations

import math
import os
import numpy as np

# ============================================================================

# ============================================================================

# Wheel ordering is shared by state vectors, commands, and kinematics.
WHEEL_ORDER = ("rear_left", "front_left", "rear_right", "front_right")


STEERING_JOINT_NAMES = [
    "rear_left_steering_joint", "front_left_steering_joint",
    "rear_right_steering_joint", "front_right_steering_joint",
]


DRIVING_JOINT_NAMES = [
    "rear_left_wheel_joint", "front_left_wheel_joint",
    "rear_right_wheel_joint", "front_right_wheel_joint",
]


WHEEL_POS = np.array([
    [-1.015,  0.510],
    [ 1.015,  0.510],
    [-1.015, -0.510],
    [ 1.015, -0.510],
], dtype=np.float32)


STEER_RANGE = {
    "rear_left":  (-math.radians(45.0), math.radians(90.0)),
    "front_left": (-math.radians(90.0), math.radians(45.0)),
    "rear_right": (-math.radians(90.0), math.radians(45.0)),
    "front_right":(-math.radians(45.0), math.radians(90.0)),
}
STEER_LIMITS = np.array([STEER_RANGE[name] for name in WHEEL_ORDER], dtype=np.float32)
STEER_SPAN = STEER_LIMITS[:, 1] - STEER_LIMITS[:, 0]


WHEEL_RADIUS = 0.155
MAX_DRIVE_VEL = 2.9
MAX_LINEAR_X = 0.45
MAX_LINEAR_Y = 0.45
MAX_ANGULAR_Z = 0.23

# ============================================================================

# ============================================================================

STEP_INTERVAL = 0.1




TRAINING_TRIGGER_MODE = "sim_time_topic_gate"
CONTROL_STEP_SIM_TIME = STEP_INTERVAL
SIM_TIME_TRIGGER_EPS = 1e-9


NUM_EPISODES = 5000


MAX_STEPS_PER_EPISODE = 20 * int(0.2 / STEP_INTERVAL)  # 40


START_RANDOM_STEPS = 500


STEERING_VEL_MAX = 1.29
DRIVING_ACC_MAX = 2.9

# ============================================================================

# ============================================================================



# ============================================================================

# ============================================================================

#   r_err = K_EXP_ERROR * exp(-||e_t|| / REWARD_ERROR_TAU)
#         + K_EXP_ERROR_FINE * exp(-||e_t|| / REWARD_ERROR_TAU_FINE)


# Dual-scale exponential tracking reward.
K_EXP_ERROR = 0.20
REWARD_ERROR_TAU = 0.45
K_EXP_ERROR_FINE = 1.20
REWARD_ERROR_TAU_FINE = 0.18


K_LATERAL_SLIP = 1.0
K_TANGENTIAL_ERROR = 1.0

# ============================================================================

# ============================================================================
STATE_DIM = 16
COMMAND_DIM = 8
GAMMA = 0.99


TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3




ENABLE_ACTOR_LR_COSINE_DECAY = True
ACTOR_LR_MIN = 5.0e-5
ACTOR_LR_DECAY_START_EPISODE = 1000
ACTOR_LR_DECAY_END_EPISODE = 8000



ENABLE_CRITIC_LR_SCHEDULE = True
CRITIC_LR_INIT = 8.0e-4
CRITIC_LR_PEAK = 1.25e-3
CRITIC_LR_MIN = 7.5e-4
CRITIC_LR_WARMUP_END_EPISODE = 1000
CRITIC_LR_HOLD_END_EPISODE = 4000
CRITIC_LR_DECAY_END_EPISODE = 8000

TARGET_POLICY_NOISE_STD = 0.03
TARGET_POLICY_NOISE_CLIP = 0.06


POLICY_UPDATE_FREQ = 6

# ============================================================================

# ============================================================================
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256



LEARNING_UPDATE_INTERVAL_STEPS = 1



CRITIC_UPDATES_PER_LEARNING_STEP = 1

# ============================================================================

# ============================================================================

EXPLORATION_NOISE_STD = 0.04


# noise_std = max(MIN_EXPLORATION_NOISE_STD,
#                 EXPLORATION_NOISE_STD - episode * EXPLORATION_NOISE_DECAY_PER_EPISODE)
EXPLORATION_NOISE_DECAY_PER_EPISODE = 1.0e-5
MIN_EXPLORATION_NOISE_STD = 0.01

# ============================================================================

# ============================================================================
# Actor rollout regularization weights and horizon.
LAMBDA_STATE_ATTRACTOR = 150.0
LAMBDA_CONTROL_MISALIGNMENT = 60.0
ROLLOUT_STEP_SIZE = 0.1
ROLLOUT_HORIZON = 15

# Actor one-step deployment/inference integration substeps.
# Keep at 1 to preserve the original policy semantics used in training/testing.
ACTOR_INTEGRATION_SUBSTEPS = 1


STATE_ATTRACTOR_STEERING_DECAY_RATE = 0.9
STATE_ATTRACTOR_WHEEL_SPEED_DECAY_RATE = 3.0


STATE_ATTRACTOR_STEERING_PROPORTION = 2.0
STATE_ATTRACTOR_WHEEL_SPEED_PROPORTION = 1.0

# ============================================================================

# ============================================================================

CONTROL_ALIGNMENT_MATRIX_MODE = "normalized"
CONTROL_ALIGNMENT_LENGTH_SCALE = float(np.linalg.norm(WHEEL_POS, axis=1).max())
CONTROL_ALIGNMENT_SPEED_SCALE = float(MAX_DRIVE_VEL)
SVD_NEAR_ZERO_THRESHOLD = 1e-6
CONDITION_PROXY_EPS = 1e-12


ENABLE_ACTOR_GRADIENT_CLIPPING = False
ACTOR_GRADIENT_CLIP_NORM = 300.0
ENABLE_ACTOR_COMPONENT_GRADIENT_LOGGING = True
ENABLE_PARAMETER_GRADIENT_LOGGING = False


STABILITY_EVENT_ACTOR_GRAD_NORM_THRESHOLD = 1.0e3
STABILITY_EVENT_CONDITION_PROXY_THRESHOLD = 1.0e6
STABILITY_EVENT_SIGMA_MIN_THRESHOLD = 1.0e-8

# ============================================================================

# ============================================================================


LEARNER_THREAD_ENABLED = True



LEARNER_REQUEST_QUEUE_MAXSIZE = 8
LEARNER_RESULT_QUEUE_MAXSIZE = 20000
LEARNER_DROP_REQUEST_WHEN_FULL = True
LEARNER_JOIN_TIMEOUT_SEC = 5.0

# ============================================================================

# ============================================================================

ACTOR_DIAGNOSTIC_INTERVAL_UPDATES = 10
ACTOR_DIAGNOSTIC_LOG_INTERVAL_UPDATES = 20
SAVE_ACTOR_UPDATE_DIAGNOSTICS = True
SAVE_NUMERICAL_STABILITY_EVENTS = True

TIMING_PROFILE_ENABLED = True
TIMING_PROFILE_PRINT_INTERVAL_EPISODES = 10

# ============================================================================

# ============================================================================

ASYNC_CSV_WRITER = True
CSV_QUEUE_MAXSIZE = 20000
CSV_FLUSH_INTERVAL_SEC = 1.0
CSV_FLUSH_EVERY_N_ROWS = 200
CSV_DROP_LOW_PRIORITY_WHEN_FULL = True


CONSOLE_PRINT_INTERVAL_STEPS = 40
CONSOLE_PRINT_INTERVAL_EPISODES = 1
CONSOLE_PRINT_REWARD_COMPONENTS = True
CONSOLE_PRINT_LOSS_COMPONENTS = True
CONSOLE_PRINT_GRADIENT_SUMMARY = True
CONSOLE_PRINT_SVD_SUMMARY = False
CONSOLE_PRINT_Q_DIAGNOSTICS = False

SAVE_STEP_REWARD_LOG = False
EPISODE_SUMMARY_LOG_INTERVAL = 1
MODEL_SAVE_INTERVAL_EPISODES = 100



ENABLE_ROLLING_CORRELATION_LOGGING = True
ROLLING_CORRELATION_WINDOW_EPISODES = 1000


# ============================================================================

# ============================================================================


# A single root seed deterministically derives component-specific seeds.
GLOBAL_RANDOM_SEED = 20260619
PYTHON_RANDOM_SEED = GLOBAL_RANDOM_SEED
NUMPY_RANDOM_SEED = GLOBAL_RANDOM_SEED + 1
TORCH_RANDOM_SEED = GLOBAL_RANDOM_SEED + 2
DOMAIN_RANDOMIZATION_BASE_SEED = GLOBAL_RANDOM_SEED + 100000



ENABLE_DOMAIN_RANDOMIZATION = True
DOMAIN_RANDOMIZATION_INTERVAL_EPISODES = 20
DOMAIN_RANDOMIZATION_READY_TIMEOUT_SEC = 15.0
DOMAIN_RESET_TOPIC = "/nvf/domain_reset_request"
DOMAIN_READY_TOPIC = "/nvf/domain_ready"


RANDOMIZE_GROUND_FRICTION = True
RANDOMIZE_PER_WHEEL_FRICTION = True
GROUND_STATIC_FRICTION_RANGE = (0.65, 0.85)
GROUND_DYNAMIC_FRICTION_RANGE = (0.60, 0.80)
PER_WHEEL_STATIC_FRICTION_MULTIPLIER_RANGE = (0.95, 1.05)
PER_WHEEL_DYNAMIC_FRICTION_MULTIPLIER_RANGE = (0.95, 1.05)


ENABLE_JOINT_SENSOR_NOISE = True
STEERING_POSITION_NOISE_STD_RAD = 0.0015
DRIVING_VELOCITY_NOISE_STD_RAD_S = 0.006
JOINT_SENSOR_NOISE_CLIP_SIGMA = 3.0


RANDOMIZE_ACTUATOR_DELAY = True
STEERING_COMMAND_DELAY_RANGE_SEC = (0.00, 0.03)
DRIVING_COMMAND_DELAY_RANGE_SEC = (0.00, 0.03)


RANDOMIZE_PAYLOAD = True
BASE_MASS_KG = 500.0
BASE_YAW_INERTIA_KG_M2 = 377.1
PAYLOAD_MASS_RANGE_KG = (0.0, 1100.0)
PAYLOAD_COM_X_RANGE_M = (-0.35, 0.35)
PAYLOAD_COM_Y_RANGE_M = (-0.22, 0.22)
PAYLOAD_COM_Z_RANGE_M = (0.15, 0.55)
PAYLOAD_YAW_INERTIA_SCALE_RANGE = (0.8, 1.25)


ENABLE_TEST_DIAGNOSTIC_CSV = False
TEST_DIAGNOSTIC_CSV_FLUSH_EVERY_N_ROWS = 100

# ============================================================================

# ============================================================================
DEFAULT_DATA_DIR = os.environ.get("NVF_DATA_DIR", "data/model")
DEFAULT_RVIZ_CONFIG = os.environ.get("NVF_RVIZ_CONFIG", "")
DEFAULT_AGV_URDF = os.environ.get("NVF_AGV_URDF", "")



ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "")
ISAAC_DEFAULT_USD_PATH = os.environ.get("NVF_ISAAC_USD_PATH", "")
ISAAC_ROS_DOMAIN_ID = 70
ISAAC_RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
ISAAC_STATUS_PUBLISH_RATE_HZ = 20.0
ISAAC_STATUS_PUBLISH_PERIOD = 1.0 / ISAAC_STATUS_PUBLISH_RATE_HZ
ISAAC_DEFAULT_MAX_SIM_RTF = 0.0


def command_velocity_limits() -> np.ndarray:
    """Return normalized actuator-state rate limits for the actor output."""
    return np.concatenate([
        2.0 * STEERING_VEL_MAX / STEER_SPAN,
        np.full(4, DRIVING_ACC_MAX / MAX_DRIVE_VEL, dtype=np.float32),
    ]).astype(np.float32)
