# Franka Isaac Sim End-Effector Tracking

This folder contains a focused SAC baseline for training a Franka arm in Isaac Sim:

- Franka 7-DoF arm driven through Isaac ROS2 topics
- Time-varying Cartesian targets: circle, figure-eight, or larger vertical figure-eight
- Reinforcement learning core: custom PyTorch SAC
- Smooth control: learned joint-acceleration policy with velocity, acceleration, and jerk limits
- Uncertainty: observation and action noise
- Metrics: tracking error, command smoothness, and success flag logged to TensorBoard

The expected Isaac topics are:

```text
/isaac_joint_states
/isaac_joint_commands
```

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Launch Isaac Sim first, confirm the topics exist with `ros2 topic list`, then run:

```bash
python -m rl_tracking.training.torch_isaac --total-timesteps 200000
```

To train on the larger vertical figure-eight:

```bash
python -m rl_tracking.training.torch_isaac \
  --trajectory horizontal8 \
  --total-timesteps 200000
```

The implementation is organized by responsibility:

```text
rl_tracking/core/        kinematics and target trajectories
rl_tracking/envs/        Isaac Gymnasium environment
rl_tracking/algorithms/  custom PyTorch SAC
rl_tracking/training/    training entry points
rl_tracking/nodes/       ROS2 policy runner
```

This saves the custom PyTorch SAC model to:

```text
runs/torch_isaac/final_model.pt
```

The PyTorch trainer owns the actor, twin critics, target critics, replay buffer, entropy temperature, and update loop.
The SAC actor outputs normalized joint acceleration commands. The controller clips acceleration, applies a jerk limit, integrates to joint velocity and joint position, then publishes a `sensor_msgs/msg/JointState` command.

## Visualize Training

The trainer writes TensorBoard events to:

```text
runs/torch_isaac/tensorboard
```

Start TensorBoard with:

```bash
tensorboard --logdir runs/torch_isaac/tensorboard --host 127.0.0.1 --port 6006
```

Then open:

```text
http://127.0.0.1:6006
```

If your controller topic is different:

```bash
python -m rl_tracking.training.torch_isaac \
  --controller-topic /isaac_joint_commands \
  --joint-states-topic /isaac_joint_states
```

Acceleration-control limits can be tuned with:

```bash
python -m rl_tracking.training.torch_isaac \
  --max-joint-speed 0.8 \
  --max-joint-accel 2.5 \
  --max-joint-jerk 18.0 \
  --residual-scale 0.35
```

## Run A Trained Policy

After training:

```bash
python -m rl_tracking.nodes.policy_runner --model runs/torch_isaac/final_model.pt
```

## Visualize The Target Trajectory

Publish the configured target path and the moving target point as ROS2 visualization markers:

```bash
python -m rl_tracking.nodes.trajectory_visualizer --trajectory figure8 --frame-id world
```

For the larger vertical figure-eight:

```bash
python -m rl_tracking.nodes.trajectory_visualizer --trajectory horizontal8 --frame-id world
```

The marker topic is:

```text
/rl_tracking/trajectory_markers
```

RViz can display this directly with a `MarkerArray` display. To show it inside the Isaac Sim viewport, your Isaac scene needs a ROS2 marker subscriber or equivalent script that converts these markers into visible USD/debug-draw geometry.

## Challenge Note

See [NOTES.md](NOTES.md) for the short design note covering state, action, reward, trajectory representation, uncertainty, and evaluation metrics.
