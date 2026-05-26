# Short Design Note

## State

The observation contains:

- 7 joint positions, normalized by joint limits
- 7 joint velocities
- current end-effector position
- desired target position and target velocity
- Cartesian tracking error
- trajectory phase as `sin(phase), cos(phase)`
- _previous command, so the policy can reduce jitter_

Explanation on `sin(phase), cos(phase)`:
It tells the RL model which section of the trajectory the desired position is at.


Gaussian observation noise is enabled during training.

## Environment And Training

The training interface follows the style:

- custom `gymnasium.Env`
- `spaces.Box` action and observation spaces
- custom PyTorch SAC actor
- twin PyTorch critics and target critics
- replay buffer
- automatic entropy temperature
- TensorBoard/CSV logger
- PyTorch checkpoints

Each episode resets the arm near a home posture with a small random joint perturbation.

Live Isaac Sim training uses `rl_tracking.training.torch_isaac`. The environment subscribes to `/isaac_joint_states`, publishes `sensor_msgs/JointState` commands to `/isaac_joint_commands`, resets the robot to a home posture at episode start, and computes the tracking reward from the live joint state.

The main trainer is `rl_tracking.training.torch_isaac`. It uses a custom PyTorch observation encoder that processes robot state, target/error state, and previous command separately before the SAC actor and critic heads.

### Two-Phase Training
The training process has two phases:
- Phase 1: trajectory tracking
In the first phase of training the rl agent focus more on guiding the end effector to stay on the given trajectory.

- Phase 2: Motion tracking
After the threshold set for trajectory satisfaction evaluation is reached, the training switches to this phase, which focus more on tracking the desired pose of the end effector at certain time. A desired position on the trajectory is being updated periodically on the trajectory, and the end effector is rewarded more if it can keep up with the desired position. 

## Uncertainty

Supported uncertainty sources:

- observation noise
- action noise

These are CLI flags in `rl_tracking.training.torch_isaac`.

## Action

The SAC policy outputs a normalized 7D joint-velocity command in `[-1, 1]`.

The simulator scales that action by the configured joint-speed limit and sends it as the commanded joint velocity. Damped least-squares IK is not part of the command path on this branch; it is only useful as a diagnostic/reference signal.

Reinforcement learning is fully responsible for choosing the robot velocity command.

## Reward

Each step rewards accurate, smooth, and safe tracking:

- Cartesian position tracking reward with an exponential bonus near the target
- velocity tracking reward: path-tangent progress in trajectory mode, target-velocity matching in timed mode
- orientation alignment reward for keeping the hand `+Z` axis near the configured target direction. This reward is quite crucial as it preempt the end effector to a single direction, which can reduce complexity
- penalty on large commanded joint velocities
- penalty on command changes between steps
- penalty near joint limits
- one-sided slow end-effector speed penalty when the hand moves below the target-speed floor
- collision penalty when contact sensors report a collision

Reward weights and penalty constants are constants in
`rl_tracking/envs/isaac.py`

The episode return is the sum of step rewards.

### Reward for Orientation
I added orientation of the end effector as a reward for two reasons. The first reason is that it would look better if the end effector is pointing downwards, as in the real world scenario it might be picking something.
The second reason is that encouraging the end effector to be pointing to a fixed direction can reduce some complexity, or reduce self-collision. This is out of instinct, I got a feeling for this and it turns out to be working perfectly.

## Trajectory

Targets are analytic functions of time:

- `horizontal8`: Lissajous-style figure-eight in the `x-y` plane, centered at the nominal home end-effector pose

The target state includes both desired position and desired velocity. Training,
deployment, and visualization all use this fixed trajectory. The center is defined
in the MoveIt/franka_description Panda base frame, `panda_link0`, near the nominal
`panda_hand` pose.

## Evaluation

Training logs:

- mean / max Cartesian tracking error
- RMS command delta as a smoothness metric
- command, policy velocity, and IK reference velocity norms
- success flag for low tracking error
- TensorBoard logs and periodic SAC checkpoints