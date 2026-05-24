# Short Design Note

## State

The observation contains:

- 7 joint positions, normalized by joint limits
- 7 joint velocities
- current end-effector position
- desired target position and target velocity
- Cartesian tracking error
- trajectory phase as `sin(phase), cos(phase)`
- previous command, so the policy can reduce jitter

Gaussian observation noise can be enabled during training.

## Environment And Training

The training interface follows the same style as the reference Gazebo project:

- custom `gymnasium.Env`
- `spaces.Box` action and observation spaces
- custom PyTorch SAC actor
- twin PyTorch critics and target critics
- replay buffer
- automatic entropy temperature
- TensorBoard/CSV logger
- PyTorch checkpoints

There is no configuration pool. Each episode resets the arm near a home posture with a small random joint perturbation.

Live Isaac Sim training uses `rl_tracking.training.torch_isaac`. The environment subscribes to `/isaac_joint_states`, publishes `sensor_msgs/JointState` commands to `/isaac_joint_commands`, resets the robot to a home posture at episode start, and computes the tracking reward from the live joint state.

The main trainer is `rl_tracking.training.torch_isaac`. It uses a custom PyTorch observation encoder that processes robot state, target/error state, and previous command separately before the SAC actor and critic heads.

## Action

The SAC policy outputs a normalized 7D joint-velocity command in `[-1, 1]`.

The simulator scales that action by the configured joint-speed limit and sends it as the commanded joint velocity. Damped least-squares IK is not part of the command path on this branch; it is only useful as a diagnostic/reference signal.

This makes reinforcement learning fully responsible for choosing the robot velocity command.

## Reward

Each step rewards accurate and smooth tracking:

- negative Cartesian position error
- penalty on large joint velocities
- penalty on command changes between steps
- penalty near joint limits
- small exponential bonus for low tracking error

Reward weights and penalty constants are module-level constants in
`rl_tracking/envs/isaac.py`, not command-line arguments.

The episode return is the sum of step rewards.

## Trajectory

Targets are analytic functions of time:

- `horizontal8`: Lissajous-style figure-eight in the `x-y` plane, centered at the nominal home end-effector pose

The target state includes both desired position and desired velocity. Training,
deployment, and visualization all use this fixed trajectory. The center is defined
in the MoveIt/franka_description Panda base frame, `panda_link0`, near the nominal
`panda_hand` pose.

## Uncertainty

Supported uncertainty sources:

- observation noise
- action noise

These are CLI flags in `rl_tracking.training.torch_isaac`.

## Evaluation

Training logs:

- mean / max Cartesian tracking error
- RMS command delta as a smoothness metric
- command, policy velocity, and IK reference velocity norms
- success flag for low tracking error
- TensorBoard logs and periodic SAC checkpoints
