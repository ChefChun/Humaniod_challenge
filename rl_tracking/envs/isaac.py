import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.control import integrate_joint_acceleration, policy_acceleration_command
from ..core.kinematics import (
    PANDA_JOINT_NAMES,
    PANDA_Q_HOME,
    PANDA_Q_MAX,
    PANDA_Q_MIN,
    damped_velocity_ik,
    forward_kinematics,
    joint_limit_cost,
    numerical_jacobian,
)
from ..core.trajectories import TrajectoryConfig, closest_target_on_trajectory, target_at


# Training-time data flow:
# 1. Isaac Sim publishes /isaac_joint_states as sensor_msgs/JointState.
# 2. Isaac contact sensors publish /collision/*; _collision_cb stores which components are unsafe.
# 3. _joint_state_cb stores those joint positions/velocities in self.q and self.qd.
# 4. make_observation packs robot state, target state, tracking error, and previous acceleration.
# 5. SAC reads that observation and outputs a normalized 7D action.
# 6. step() treats that action as the main joint acceleration command, integrates to velocity/position,
#    then publishes /isaac_joint_commands back to Isaac Sim.
def make_observation(
    q: np.ndarray,
    qd: np.ndarray,
    prev_action: np.ndarray,
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    phase: float,
    noise_std: float = 0.0,
    prev_action_scale: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    q_center = 0.5 * (PANDA_Q_MAX + PANDA_Q_MIN)
    q_halfspan = 0.5 * (PANDA_Q_MAX - PANDA_Q_MIN)
    # Keep this ordering aligned with TrackingEncoder in algorithms/sac.py:
    # [normalized joints, joint velocities, EE position, target position,
    #  target velocity, target error, trajectory phase, previous acceleration].
    obs = np.concatenate(
        [
            (q - q_center) / q_halfspan,
            qd / 2.0,
            ee_pos,
            target_pos,
            target_vel,
            target_pos - ee_pos,
            np.array([np.sin(phase), np.cos(phase)]),
            prev_action / prev_action_scale,
        ]
    ).astype(float)

    if noise_std > 0.0 and rng is not None:
        obs = obs + rng.normal(0.0, noise_std, size=obs.shape)
    return obs


@dataclass
class IsaacEnvConfig:
    dt: float = 0.08
    horizon: int = 180
    trajectory: str = "figure8"
    obs_noise: float = 0.001
    action_noise: float = 0.01
    action_accel_scale: float = 1.0
    max_joint_speed: float = 0.8
    max_joint_accel: float = 2.5
    max_joint_jerk: float = 18.0
    trajectory_projection_samples: int = 180
    controller_topic: str = "/isaac_joint_commands"
    joint_states_topic: str = "/isaac_joint_states"
    collision_topic: str = "/collision"
    collision_topics: tuple[str, ...] = ()
    collision_msg_type: str = "std_msgs/msg/Bool"
    collision_threshold: float = 0.0
    collision_penalty: float = 20.0
    terminate_on_collision: bool = True
    reset_duration: float = 2.0
    command_duration: float = 0.12
    settle_timeout: float = 20.0
    seed: int = 7


def _load_ros_message_type(type_name: str):
    normalized = type_name.replace(".", "/")
    if normalized == "Bool":
        normalized = "std_msgs/msg/Bool"
    try:
        from rosidl_runtime_py.utilities import get_message

        return get_message(normalized)
    except Exception:
        if normalized == "std_msgs/msg/Bool":
            from std_msgs.msg import Bool

            return Bool
        if normalized == "std_msgs/msg/Float32":
            from std_msgs.msg import Float32

            return Float32
        if normalized == "std_msgs/msg/Float64":
            from std_msgs.msg import Float64

            return Float64
        if normalized == "geometry_msgs/msg/WrenchStamped":
            from geometry_msgs.msg import WrenchStamped

            return WrenchStamped
        raise


def _collision_component_name(topic: str) -> str:
    normalized = topic.rstrip("/")
    prefix = "/collision/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :].replace("/", ".")
    return normalized.rsplit("/", 1)[-1] or normalized


class IsaacFrankaTrackingEnv(gym.Env):
    """Gymnasium environment that trains the Franka by commanding Isaac Sim over ROS2."""

    metadata = {"render_modes": []}
    action_dim = 7
    observation_dim = 35

    def __init__(self, config: IsaacEnvConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        # This node is the bridge between Gym/SAC Python code and Isaac's ROS2 topics.
        self._rclpy = rclpy
        self._JointState = JointState
        self._node = rclpy.create_node(f"isaac_franka_tracking_env_{id(self)}")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        collision_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._command_pub = self._node.create_publisher(
            JointState,
            config.controller_topic,
            qos,
        )
        self._joint_sub = self._node.create_subscription(
            JointState,
            config.joint_states_topic,
            self._joint_state_cb,
            qos,
        )
        self._configured_collision_topics = tuple(config.collision_topics)
        self._collision_root_topic = config.collision_topic.rstrip("/")
        self._collision_qos = collision_qos
        self._collision_msg_cls = _load_ros_message_type(config.collision_msg_type)
        self._collision_subs = {}
        self.collision_topics: tuple[str, ...] = ()
        self.collision_components: tuple[str, ...] = ()
        self.collision_states: dict[str, bool] = {}
        self.collision_magnitudes: dict[str, float] = {}
        self.collision_event_counts: dict[str, int] = {}
        self.in_collision = False
        self.collision_magnitude = 0.0
        self.collision_count = 0
        self._refresh_collision_subscriptions()

        self.q: np.ndarray | None = None
        self.qd = np.zeros(7, dtype=float)
        self.command_velocity = np.zeros(7, dtype=float)
        self.prev_acceleration = np.zeros(7, dtype=float)
        self.reward_mode = "trajectory"
        self.tracking_start_time: float | None = None
        self.step_count = 0
        self.t = 0.0
        self.trajectory_cfg = TrajectoryConfig(kind=config.trajectory)
        self._wait_for_joint_state()

    def _refresh_collision_subscriptions(self) -> None:
        if self._node is None:
            return
        if self._configured_collision_topics:
            topics = self._configured_collision_topics
        else:
            topics = sorted(
                name
                for name, _ in self._node.get_topic_names_and_types()
                if name.startswith(f"{self._collision_root_topic}/")
            )
            if not topics and not self._collision_subs:
                topics = (self.config.collision_topic,)

        for topic in topics:
            self._subscribe_collision_topic(topic)

    def _subscribe_collision_topic(self, topic: str) -> None:
        if topic in self._collision_subs:
            return
        component = _collision_component_name(topic)
        self.collision_states.setdefault(component, False)
        self.collision_magnitudes.setdefault(component, 0.0)
        self.collision_event_counts.setdefault(component, 0)
        self._collision_subs[topic] = self._node.create_subscription(
            self._collision_msg_cls,
            topic,
            lambda msg, component=component: self._collision_cb(msg, component),
            self._collision_qos,
        )
        self.collision_topics = tuple(self._collision_subs.keys())
        self.collision_components = tuple(_collision_component_name(topic) for topic in self.collision_topics)
        self._node.get_logger().info(f"Collision monitor subscribed to {topic} as component '{component}'")

    def _visible_collision_component_topics(self) -> list[str]:
        if self._node is None:
            return []
        return sorted(
            name
            for name, _ in self._node.get_topic_names_and_types()
            if name.startswith(f"{self._collision_root_topic}/")
        )

    def _joint_state_cb(self, msg) -> None:
        # ROS2 JointState arrays are name-indexed; rebuild q/qd in the fixed Panda joint order.
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if not all(name in positions for name in PANDA_JOINT_NAMES):
            return
        self.q = np.array([positions[name] for name in PANDA_JOINT_NAMES], dtype=float)
        self.qd = np.array([velocities.get(name, 0.0) for name in PANDA_JOINT_NAMES], dtype=float)

    def _collision_cb(self, msg, component: str) -> None:
        # Each collision topic owns one component state, e.g. /collision/hand -> hand.
        # Bool topics are expected, but numeric/contact-force messages are also accepted.
        in_collision, magnitude = self._parse_collision_message(msg)
        was_in_collision = self.collision_states.get(component, False)
        self.collision_states[component] = in_collision
        self.collision_magnitudes[component] = magnitude
        if in_collision:
            self.collision_event_counts[component] = self.collision_event_counts.get(component, 0) + 1
            self.collision_count += 1
        self.in_collision = any(self.collision_states.values())
        self.collision_magnitude = max(self.collision_magnitudes.values(), default=0.0)
        if in_collision and not was_in_collision and self._node is not None:
            active = ", ".join(self._active_collision_components())
            self._node.get_logger().warn(f"Collision detected on {component}; active components: {active}")

    def _active_collision_components(self) -> list[str]:
        return sorted(component for component, active in self.collision_states.items() if active)

    def _parse_collision_message(self, msg) -> tuple[bool, float]:
        if hasattr(msg, "data"):
            data = msg.data
            if isinstance(data, bool):
                return data, 1.0 if data else 0.0
            magnitude = abs(float(data))
            return magnitude > self.config.collision_threshold, magnitude

        if hasattr(msg, "wrench"):
            force = msg.wrench.force
            torque = msg.wrench.torque
            magnitude = float(
                np.linalg.norm([force.x, force.y, force.z]) + np.linalg.norm([torque.x, torque.y, torque.z])
            )
            return magnitude > self.config.collision_threshold, magnitude

        if hasattr(msg, "force"):
            force = msg.force
            magnitude = float(np.linalg.norm([force.x, force.y, force.z]))
            return magnitude > self.config.collision_threshold, magnitude

        if hasattr(msg, "contacts"):
            count = float(len(msg.contacts))
            return count > self.config.collision_threshold, count

        if hasattr(msg, "states"):
            count = float(len(msg.states))
            return count > self.config.collision_threshold, count

        return False, 0.0

    def _spin_for(self, duration: float) -> None:
        deadline = time.time() + duration
        if self._node is not None:
            while time.time() < deadline:
                self._refresh_collision_subscriptions()
                self._rclpy.spin_once(self._node, timeout_sec=0.01)

    def _wait_for_joint_state(self) -> None:
        start = time.time()
        if self._node is not None:
            while self.q is None:
                self._refresh_collision_subscriptions()
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
                if time.time() - start > self.config.settle_timeout:
                    topics = self._node.get_topic_names_and_types()
                    topic_names = sorted(name for name, _ in topics)
                    raise RuntimeError(
                        "Timed out waiting for Franka joint states. "
                        f"Check that Isaac Sim publishes {self.config.joint_states_topic} "
                        f"with joints {PANDA_JOINT_NAMES}. "
                        f"ROS topics visible to this node: {topic_names}. "
                    )

    def _publish_position_command(self, q_desired: np.ndarray, qd_desired: np.ndarray | None = None) -> None:
        if self._node is not None:
            # Isaac consumes desired joint position and optional velocity on the command topic.
            command = self._JointState()
            command.header.stamp = self._node.get_clock().now().to_msg()
            command.name = PANDA_JOINT_NAMES
            command.position = np.clip(q_desired, PANDA_Q_MIN, PANDA_Q_MAX).tolist()
            if qd_desired is not None:
                command.velocity = np.asarray(qd_desired, dtype=float).tolist()
            self._command_pub.publish(command)

    def _target_time(self) -> float:
        # Strict tracking starts its own trajectory clock when the trainer switches modes.
        if self.reward_mode == "timed" and self.tracking_start_time is not None:
            return max(0.0, self.t - self.tracking_start_time)
        return self.t

    def _target(self) -> tuple[np.ndarray, np.ndarray, float]:
        return target_at(self._target_time(), self.trajectory_cfg)

    def set_reward_mode(self, mode: str) -> None:
        if mode not in {"trajectory", "timed"}:
            raise ValueError(f"Unknown reward mode: {mode}")
        if self.reward_mode != mode and mode == "timed":
            self.tracking_start_time = self.t
        elif mode == "trajectory":
            self.tracking_start_time = None
        self.reward_mode = mode

    def _ee_position(self) -> np.ndarray:
        assert self.q is not None
        return forward_kinematics(self.q)

    def _observe(self) -> np.ndarray:
        assert self.q is not None
        ee_pos = self._ee_position()
        if self.reward_mode == "trajectory":
            target_pos, target_vel, phase, _ = closest_target_on_trajectory(
                ee_pos,
                self.trajectory_cfg,
                samples=self.config.trajectory_projection_samples,
            )
        else:
            target_pos, target_vel, phase = self._target()
        obs = make_observation(
            self.q,
            self.qd,
            self.prev_acceleration,
            ee_pos,
            target_pos,
            target_vel,
            phase,
            noise_std=self.config.obs_noise,
            prev_action_scale=self.config.max_joint_accel,
            rng=self.rng,
        )
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.t = 0.0
        self.command_velocity = np.zeros(7, dtype=float)
        self.prev_acceleration = np.zeros(7, dtype=float)
        self.collision_states = {component: False for component in self.collision_components}
        self.collision_magnitudes = {component: 0.0 for component in self.collision_components}
        self.collision_event_counts = {component: 0 for component in self.collision_components}
        self.in_collision = any(self.collision_states.values())
        self.collision_magnitude = max(self.collision_magnitudes.values(), default=0.0)
        self.collision_count = 0
        self.tracking_start_time = 0.0 if self.reward_mode == "timed" else None
        self.trajectory_cfg = TrajectoryConfig(kind=self.config.trajectory)

        random_offset = self.rng.normal(0.0, 0.025, size=7)
        reset_q = np.clip(PANDA_Q_HOME + random_offset, PANDA_Q_MIN, PANDA_Q_MAX)
        self._publish_position_command(reset_q, np.zeros(7))
        self._spin_for(self.config.reset_duration)
        self._wait_for_joint_state()
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.q is not None
        collision_count_before = self.collision_count
        collision_event_counts_before = dict(self.collision_event_counts)
        collision_components_before = set(self._active_collision_components())
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.config.action_noise > 0.0:
            action = np.clip(action + self.rng.normal(0.0, self.config.action_noise, size=7), -1.0, 1.0)

        ee_pos = self._ee_position()
        if self.reward_mode == "trajectory":
            target_pos, target_vel, _, _ = closest_target_on_trajectory(
                ee_pos,
                self.trajectory_cfg,
                samples=self.config.trajectory_projection_samples,
            )
        else:
            target_pos, target_vel, _ = self._target()
        error_vec = target_pos - ee_pos
        # IK is diagnostic/reference only on this branch; SAC supplies the actual acceleration command.
        reference_velocity = damped_velocity_ik(
            self.q,
            error_vec,
            target_vel,
            max_joint_speed=self.config.max_joint_speed,
        )
        desired_acceleration = policy_acceleration_command(
            action,
            self.config.max_joint_accel,
            self.config.action_accel_scale,
        )
        # The command sent to Isaac is still JointState(position, velocity); acceleration is
        # an internal control signal integrated over dt to get those publishable quantities.
        desired_q, command_velocity, acceleration, acceleration_delta = integrate_joint_acceleration(
            self.q,
            self.command_velocity,
            self.prev_acceleration,
            desired_acceleration,
            self.config.dt,
            self.config.max_joint_speed,
            self.config.max_joint_accel,
            self.config.max_joint_jerk,
        )
        self._publish_position_command(desired_q, command_velocity)
        self._spin_for(self.config.dt)

        self.command_velocity = command_velocity
        self.t += self.config.dt
        self.step_count += 1

        ee_after = self._ee_position()
        closest_target_pos, closest_target_vel, _, trajectory_error = closest_target_on_trajectory(
            ee_after,
            self.trajectory_cfg,
            samples=self.config.trajectory_projection_samples,
        )
        ee_velocity = numerical_jacobian(self.q) @ self.qd

        next_target_pos, next_target_vel, _ = self._target()

        if self.reward_mode == "trajectory":
            reward_target_pos = closest_target_pos
            reward_target_vel = closest_target_vel
            error = trajectory_error
            to_path = closest_target_pos - ee_after
            to_path_norm = float(np.linalg.norm(to_path))
            if to_path_norm > 1e-6:
                velocity_toward_path = float(np.dot(ee_velocity, to_path / to_path_norm))
            else:
                velocity_toward_path = 0.0
            tangent_norm = float(np.linalg.norm(closest_target_vel))
            velocity_along_trajectory = float(np.dot(ee_velocity, closest_target_vel / tangent_norm)) if tangent_norm > 1e-6 else 0.0
            velocity_reward = (
                0.8 * float(np.clip(velocity_toward_path, -0.3, 0.3))
                + 1.4 * float(np.clip(velocity_along_trajectory, -0.3, 0.3))
            )
            position_reward = -6.0 * error + 0.30 * float(np.exp(-45.0 * error))
        else:
            reward_target_pos = next_target_pos
            reward_target_vel = next_target_vel
            error = float(np.linalg.norm(next_target_pos - ee_after))
            velocity_error = float(np.linalg.norm(ee_velocity - next_target_vel))
            velocity_toward_path = 0.0
            velocity_along_trajectory = 0.0
            velocity_reward = -1.2 * velocity_error + 0.25 * float(np.exp(-8.0 * velocity_error))
            position_reward = -7.0 * error + 0.45 * float(np.exp(-35.0 * error))

        smoothness = float(np.linalg.norm(acceleration_delta))
        jerk_norm = smoothness / self.config.dt
        limit_penalty = joint_limit_cost(self.q)
        collision_events_this_step = self.collision_count - collision_count_before
        collision_event_components = {
            component
            for component, count in self.collision_event_counts.items()
            if count > collision_event_counts_before.get(component, 0)
        }
        collision_components_this_step = sorted(
            collision_components_before | set(self._active_collision_components()) | collision_event_components
        )
        step_collision = bool(collision_components_this_step)
        collision_penalty = self.config.collision_penalty if step_collision else 0.0

        # Two-phase reward:
        # trajectory mode: stay near the geometric path and move along its tangent;
        # timed mode: after trainer-level curriculum switch, follow the time-indexed target.
        reward = (
            position_reward
            + velocity_reward
            - 0.01 * float(np.dot(command_velocity, command_velocity))
            - 0.015 * float(np.dot(acceleration, acceleration))
            - 0.05 * smoothness
            - 6.0 * limit_penalty
            - collision_penalty
        )

        self.prev_acceleration = acceleration.copy()
        terminated = self.config.terminate_on_collision and step_collision
        truncated = self.step_count >= self.config.horizon
        info = {
            "time": self.t,
            "ee_pos": ee_after,
            "target_pos": next_target_pos,
            "reward_target_pos": reward_target_pos,
            "reward_target_vel": reward_target_vel,
            "closest_target_pos": closest_target_pos,
            "error": error,
            "timed_error": float(np.linalg.norm(next_target_pos - ee_after)),
            "trajectory_error": trajectory_error,
            "velocity_toward_path": velocity_toward_path,
            "velocity_along_trajectory": velocity_along_trajectory,
            "velocity_reward": velocity_reward,
            "position_reward": position_reward,
            "ee_velocity": ee_velocity,
            "smoothness": smoothness,
            "jerk_norm": jerk_norm,
            "reference_velocity": reference_velocity,
            "acceleration": acceleration,
            "command_velocity": command_velocity,
            "command": command_velocity,
            "in_collision": step_collision,
            "collision_active": self.in_collision,
            "collision_components": collision_components_this_step,
            "collision_magnitude": self.collision_magnitude,
            "collision_events_this_step": collision_events_this_step,
            "collision_count": self.collision_count,
            "collision_penalty": collision_penalty,
            "is_success": error < 0.035,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
