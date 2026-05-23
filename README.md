# Franka Isaac Sim End-Effector Tracking

This folder contains a focused SAC baseline for training a Franka arm in Isaac Sim:

- Franka 7-DoF arm driven through Isaac ROS2 topics
- Time-varying Cartesian targets: circle, figure-eight, or larger horizontal figure-eight
- Reinforcement learning core: custom PyTorch SAC
- Smooth control: learned joint-acceleration policy with velocity, acceleration, and jerk limits
- Safety: optional collision penalty from Isaac contact sensors on `/collision/*`
- Uncertainty: observation and action noise
- Metrics: tracking error, end-effector speed, orientation alignment, command smoothness, and success flag logged to TensorBoard

The expected Isaac topics are:

```text
/isaac_joint_states
/isaac_joint_commands
/collision/hand
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

To train on the larger horizontal figure-eight:

```bash
python -m rl_tracking.training.torch_isaac \
  --trajectory horizontal8 \
  --total-timesteps 200000
```

The trajectory kind, center, radius, period, and unreachable stress segment are
saved in `isaac_env_config.json` and reused by the policy runner:

```bash
python -m rl_tracking.training.torch_isaac \
  --trajectory figure8 \
  --trajectory-center 0.4174 0.0 0.4558 \
  --trajectory-radius 0.08 \
  --trajectory-period 6.0
```

By default, training auto-subscribes to visible `/collision/...` component topics as
`std_msgs/msg/Bool`, falling back to `/collision` if no component topics are visible yet.
You can also repeat `--collision-topic` to set the list explicitly. A collision subtracts
`--collision-penalty` from the reward, logs the component name, and terminates the current episode:

```bash
python -m rl_tracking.training.torch_isaac \
  --collision-topic /collision/hand \
  --collision-topic /collision/left_finger \
  --collision-topic /collision/right_finger \
  --collision-msg-type std_msgs/msg/Bool \
  --collision-penalty 20.0
```

Use `--no-terminate-on-collision` if you want collisions to reduce reward without ending the episode.

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

The reward includes a small end-effector direction term. By default it encourages
the Panda hand-frame `+Z` axis to stay aligned with the home pose direction,
which is approximately base-frame `-Z`, while the position and velocity tracking
terms remain dominant:

```bash
python -m rl_tracking.training.torch_isaac \
  --orientation-reward-weight 0.15 \
  --orientation-target-direction 0 0 -1
```

The reward also includes a one-sided slow-motion penalty. The end effector is
penalized only when its speed falls below one fifth of the current target
trajectory speed; moving faster than that floor does not add reward:

```bash
python -m rl_tracking.training.torch_isaac \
  --min-ee-speed-fraction 0.2 \
  --slow-speed-penalty-weight 2.0
```

## Run A Trained Policy

After training:

```bash
python -m rl_tracking.nodes.policy_runner --model runs/torch_isaac/final_model.pt
```

## Run A Direct IK Trajectory Test

To test the trajectory and command topics without RL, run the kinematic runner:

```bash
python -m rl_tracking.nodes.kinematic_runner --trajectory horizontal8
```

For the MoveIt Panda model, the default trajectory center is in `panda_link0`
coordinates near the nominal `panda_hand` pose. The horizontal figure-eight keeps
`z` constant at the configured center height and starts at that center. Its long
axis runs side-to-side in the horizontal plane. The current horizontal8 size is
smaller than the previous large version while keeping the same center.
The kinematic runner first moves to the nearest point on the path, then starts
advancing along the trajectory.

## Visualize The Target Trajectory

Publish the configured target path and the moving target point as ROS2 visualization markers:

```bash
python -m rl_tracking.nodes.trajectory_visualizer --trajectory figure8 --frame-id panda_link0
```

For the larger horizontal figure-eight:

```bash
python -m rl_tracking.nodes.trajectory_visualizer --trajectory horizontal8 --frame-id panda_link0
```

Use the same `--center`, `--radius`, `--period`, and `--unreachable` values here
when visualizing a custom training trajectory.
The green end-effector marker uses TF from `panda_link0` to `panda_hand` by
default and falls back to the local FK estimate if TF is unavailable. If your
Isaac scene does not publish TF, add a ROS2 Publish Transform Tree node for
`/Franka` or run the visualizer with `--ee-source fk`.
The orange moving target marker has its own clock; the kinematic runner may not
be phase-synchronized with it because it starts from the nearest path point.
Use the blue path for geometric tracking checks, or run the runner with
`--start-mode fixed --approach-duration 0` when you need phase-zero comparison.

The marker topic is:

```text
/rl_tracking/trajectory_markers
```

RViz can display this directly with a `MarkerArray` display. To show it inside the Isaac Sim viewport, your Isaac scene needs a ROS2 marker subscriber or equivalent script that converts these markers into visible USD/debug-draw geometry.

## Challenge Note

See [NOTES.md](NOTES.md) for the short design note covering state, action, reward, trajectory representation, uncertainty, and evaluation metrics.
