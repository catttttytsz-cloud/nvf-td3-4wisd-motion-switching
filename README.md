# Motion Mode Switching Control in 4WISD Robots Based on Neural Vector Fields

This repository contains the training implementation associated with the paper “Motion Mode Switching Control in 4WISD Robots based on Neural Vector Field with State Trajectory Regularization.” The code is to facilitate understanding and reproduction of the training procedure, as well as reuse by the community, under the terms of the MIT License.

The method uses TD3 for motion-mode switching control of a four-wheel-independent-steering-and-driving (4WISD) robot. The policy uses a neural vector field actor with state-attractor and control-misalignment trajectory regularization.

## Current limitation

The robot model used in the paper cannot be released because of intellectual-property restrictions. You may use your own 4WISD robot model or adapt the open-source [xmobot](https://github.com/YeatsWang/xmobot) model. Robot dimensions, joint definitions, actuator limits, Isaac Sim prim paths, and the 4WISD kinematic parameters in this repository must be adjusted to match the selected model. The current xmobot project targets ROS 1 Noetic and Gazebo, so using it with this ROS 2 and Isaac Sim training pipeline requires additional integration.

## Environment

Create the Conda environment supplied with the repository:

```bash
conda env create -f env/environment.yml
conda activate rl_env
```

Build the ROS 2 package from the workspace root and source it:

```bash
colcon build --packages-select neural_vector_field
source install/setup.bash
```

## Training

Set the Isaac Sim installation path and start the simulator bridge with the USD file for your adapted 4WISD robot.

For headless accelerated training:

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
bash scripts/run_isaac_headless_training.sh /path/to/your_4wisd_scene.usd
```

For training with the Isaac Sim UI:

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
bash scripts/run_isaac_ui_velocity_training.sh /path/to/your_4wisd_scene.usd
```

The optional second and third script arguments are the state publication rate and maximum simulation real-time factor. Their defaults are `20` Hz and `0` (unlimited, but not recommended; simulation and training rtf are supposed to be equal as much).

In another terminal, activate the Conda environment, source the ROS 2 workspace, provide the URDF and optional RViz configuration, and launch training:

```bash
conda activate rl_env
source install/setup.bash
export NVF_AGV_URDF=/path/to/your_robot.urdf
export NVF_RVIZ_CONFIG=/path/to/your_config.rviz
ros2 launch neural_vector_field train_vector_field_policy.launch.py
```

RViz and live plots can be disabled when they are not needed:

```bash
ros2 launch neural_vector_field train_vector_field_policy.launch.py \
  launch_rviz:=false launch_plot:=false
```

## Configuration

The main training configuration is `neural_vector_field/config.py`. Before training with a different robot, update at least the wheel order, joint names, wheel positions, steering limits, wheel radius, velocity limits, and any model-specific paths. Review `neural_vector_field/four_wisd_kinematics.py` and adapt the kinematic equations if the selected robot uses a different coordinate convention or wheel arrangement.

The primary implementation files are:

- `neural_vector_field/vector_field_actor.py`: neural vector field actor and trajectory regularization;
- `neural_vector_field/td3_vector_field_agent.py`: TD3 agent and replay buffer;
- `neural_vector_field/train_vector_field_policy.py`: ROS 2 training node;
- `neural_vector_field/config.py`: robot and training parameters;
- `scripts/isaac_agv_state_publisher_headless.py`: headless Isaac Sim bridge;
- `scripts/isaac_agv_state_publisher_ui_velocity.py`: Isaac Sim UI bridge.



## License

This project is released under the [MIT License](LICENSE).


