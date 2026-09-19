#!/usr/bin/env bash
set -euo pipefail

USD_PATH="${1:?Usage: $0 /absolute/path/to/training_world.usd [publish_rate_hz] [max_sim_rtf]}"
PUBLISH_RATE_HZ="${2:-20}"
MAX_SIM_RTF="${3:-0}"

unset PYTHONPATH
unset PYTHONHOME
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=70

isaac_sim_package_path="${ISAAC_SIM_PATH:?Set ISAAC_SIM_PATH to the Isaac Sim installation directory}"
export isaac_sim_package_path
export LD_LIBRARY_PATH="$isaac_sim_package_path/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$isaac_sim_package_path"

./python.sh "$SCRIPT_DIR/isaac_agv_state_publisher_ui_velocity.py" \
  --usd "$USD_PATH" \
  --publish-rate-hz "$PUBLISH_RATE_HZ" \
  --max-sim-rtf "$MAX_SIM_RTF" \
  --/app/runLoops/main/rateLimitEnabled=false \
  --/app/runLoops/present/rateLimitEnabled=false \
  --/app/runLoops/rendering_0/rateLimitEnabled=false
