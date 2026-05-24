import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.control import integrate_joint_velocity, policy_velocity_command
from ..core.kinematics import (
    PANDA_HOME_EE_DIRECTION,
    PANDA_JOINT_NAMES,
    PANDA_Q_HOME,
    PANDA_Q_MAX,
    PANDA_Q_MIN,
    damped_velocity_ik,
    forward_kinematics,
    hand_z_axis,
    joint_limit_cost,
    numerical_jacobian,
)
from ..core.trajectories import (
    DEFAULT_TRAJECTORY_CENTER,
    DEFAULT_TRAJECTORY_PERIOD,
    DEFAULT_TRAJECTORY_RADIUS,
    TrajectoryConfig,
    closest_target_on_trajectory_time,
    make_trajectory_config,
    target_at,
)


# Training-time data flow:
# 1. Isaac Sim publishes /isaac_joint_states as sensor_msgs/JointState.
# 2. Isaac contact sensors publish /collision/*; _collision_cb stores which components are unsafe.
# 3. _joint_state_cb stores those joint positions/velocities in self.q and self.qd.
# 4. make_observation packs robot state, target state, tracking error, and previous velocity command.
# 5. SAC reads that observation and outputs a normalized 7D action.
# 6. step() treats that action as the full joint-velocity command, then publishes
#    /isaac_joint_commands back to Isaac Sim.
def make_observation(
    q: np.ndarray,
    qd: np.ndarray,
    prev_command: np.ndarray,
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    phase: float,
    noise_std: float = 0.0,
    prev_command_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    q_center = 0.5 * (PANDA_Q_MAX + PANDA_Q_MIN)
    q_halfspan = 0.5 * (PANDA_Q_MAX - PANDA_Q_MIN)
    prev_command_scale = max(float(prev_command_scale), 1e-6)
    # Keep this ordering aligned with TrackingEncoder in algorithms/sac.py:
    # [normalized joints, joint velocities, EE position, target position,
    #  target velocity, target error, trajectory phase, previous velocity command].
    obs = np.concatenate(
        [
            (q - q_center) / q_halfspan,
            qd / 2.0,
            ee_pos,
            target_pos,
            target_vel,
            target_pos - ee_pos,
            np.array([np.sin(phase), np.cos(phase)]),
            prev_command / prev_command_scale,
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
    trajectory_center: tuple[float, float, float] = DEFAULT_TRAJECTORY_CENTER
    trajectory_radius: float = DEFAULT_TRAJECTORY_RADIUS
    trajectory_period: float = DEFAULT_TRAJECTORY_PERIOD
    trajectory_unreachable: bool = False
    obs_noise: float = 0.001
    action_noise: float = 0.01
    action_velocity_scale: float = 1.0
    max_joint_speed: float = 0.8
    orientation_reward_weight: float = 0.15
    orientation_target_direction: tuple[float, float, float] = PANDA_HOME_EE_DIRECTION
    min_ee_speed_fraction: float = 0.2
    slow_speed_penalty_weight: float = 2.0
    smoothness_penalty_weight: float = 0.12
    phase_two_smoothness_penalty_weight: float = 0.24
    trajectory_projection_samples: int = 180
    trajectory_projection_window: float = 1.2
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


def _normalize_direction(direction: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    direction_arr = np.asarray(direction, dtype=float)
    if direction_arr.shape != (3,):
        raise ValueError(f"Expected 3D orientation target direction, got shape {direction_arr.shape}")
    norm = float(np.linalg.norm(direction_arr))
    if norm < 1e-9:
        raise ValueError("Orientation target direction must have non-zero length")
    return direction_arr / norm


def _speed_floor_penalty(
    ee_velocity: np.ndarray,
    expected_velocity: np.ndarray,
    min_speed_fraction: float,
    penalty_weight: float,
) -> tuple[float, float, float, float]:
    ee_speed = float(np.linalg.norm(ee_velocity))
    expected_speed = float(np.linalg.norm(expected_velocity))
    min_expected_speed = min_speed_fraction * expected_speed
    speed_deficit = max(0.0, min_expected_speed - ee_speed)
    return penalty_weight * speed_deficit, ee_speed, expected_speed, min_expected_speed


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
        self.prev_command = np.zeros(7, dtype=float)
        self.reward_mode = "trajectory"
        self.path_time_estimate: float | None = None
        self.step_count = 0
        self.t = 0.0
        self.trajectory_cfg = self._make_trajectory_config()
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if config.max_joint_speed <= 0.0:
            raise ValueError("max_joint_speed must be positive")
        if config.action_velocity_scale < 0.0:
            raise ValueError("action_velocity_scale must be non-negative")
        if config.orientation_reward_weight < 0.0:
            raise ValueError("orientation_reward_weight must be non-negative")
        if config.min_ee_speed_fraction < 0.0:
            raise ValueError("min_ee_speed_fraction must be non-negative")
        if config.slow_speed_penalty_weight < 0.0:
            raise ValueError("slow_speed_penalty_weight must be non-negative")
        if config.smoothness_penalty_weight < 0.0:
            raise ValueError("smoothness_penalty_weight must be non-negative")
        if config.phase_two_smoothness_penalty_weight < 0.0:
            raise ValueError("phase_two_smoothness_penalty_weight must be non-negative")
        self.orientation_target_direction = _normalize_direction(config.orientation_target_direction)
        self._wait_for_joint_state()

    def _make_trajectory_config(self) -> TrajectoryConfig:
        return make_trajectory_config(
            kind=self.config.trajectory,
            center=self.config.trajectory_center,
            radius=self.config.trajectory_radius,
            period=self.config.trajectory_period,
            unreachable=self.config.trajectory_unreachable,
        )

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

    def _closest_path_target_with_time(
        self,
        ee_pos: np.ndarray,
        *,
        update_estimate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        if self.path_time_estimate is None:
            target_pos, target_vel, phase, distance, target_time = closest_target_on_trajectory_time(
                ee_pos,
                self.trajectory_cfg,
                samples=self.config.trajectory_projection_samples,
            )
        else:
            target_pos, target_vel, phase, distance, target_time = closest_target_on_trajectory_time(
                ee_pos,
                self.trajectory_cfg,
                samples=self.config.trajectory_projection_samples,
                center_time=self.path_time_estimate,
                search_window=self.config.trajectory_projection_window,
            )
        if update_estimate:
            self.path_time_estimate = target_time
        return target_pos, target_vel, phase, distance, target_time

    def _closest_path_target(
        self,
        ee_pos: np.ndarray,
        *,
        update_estimate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        target_pos, target_vel, phase, distance, _ = self._closest_path_target_with_time(
            ee_pos,
            update_estimate=update_estimate,
        )
        return target_pos, target_vel, phase, distance

    def _next_path_target_from_position(
        self,
        ee_pos: np.ndarray,
        *,
        update_estimate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        _, _, _, distance, target_time = self._closest_path_target_with_time(
            ee_pos,
            update_estimate=update_estimate,
        )
        target_pos, target_vel, phase = target_at(target_time + self.config.dt, self.trajectory_cfg)
        return target_pos, target_vel, phase, distance

    def _desired_target(
        self,
        ee_pos: np.ndarray,
        *,
        update_estimate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self.reward_mode == "trajectory":
            return self._closest_path_target(ee_pos, update_estimate=update_estimate)
        return self._next_path_target_from_position(ee_pos, update_estimate=update_estimate)

    def set_reward_mode(self, mode: str) -> None:
        if mode not in {"trajectory", "timed"}:
            raise ValueError(f"Unknown reward mode: {mode}")
        if self.reward_mode != mode and mode == "timed":
            if self.q is not None:
                self.path_time_estimate = None
                self._next_path_target_from_position(self._ee_position(), update_estimate=True)
            if self._node is not None:
                self._node.get_logger().info(
                    "Training phase two enabled: target is now the next trajectory point "
                    "projected from the current end-effector position."
                )
        elif mode == "trajectory":
            self.path_time_estimate = None
        self.reward_mode = mode

    def _ee_position(self) -> np.ndarray:
        assert self.q is not None
        return forward_kinematics(self.q)

    def _observe(self) -> np.ndarray:
        assert self.q is not None
        ee_pos = self._ee_position()
        target_pos, target_vel, phase, _ = self._desired_target(ee_pos)
        obs = make_observation(
            self.q,
            self.qd,
            self.prev_command,
            ee_pos,
            target_pos,
            target_vel,
            phase,
            noise_std=self.config.obs_noise,
            prev_command_scale=self.config.max_joint_speed,
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
        self.prev_command = np.zeros(7, dtype=float)
        self.collision_states = {component: False for component in self.collision_components}
        self.collision_magnitudes = {component: 0.0 for component in self.collision_components}
        self.collision_event_counts = {component: 0 for component in self.collision_components}
        self.in_collision = any(self.collision_states.values())
        self.collision_magnitude = max(self.collision_magnitudes.values(), default=0.0)
        self.collision_count = 0
        self.trajectory_cfg = self._make_trajectory_config()
        self.path_time_estimate = None

        random_offset = self.rng.normal(0.0, 0.025, size=7)
        reset_q = np.clip(PANDA_Q_HOME + random_offset, PANDA_Q_MIN, PANDA_Q_MAX)
        self._publish_position_command(reset_q, np.zeros(7))
        self._spin_for(self.config.reset_duration)
        self._wait_for_joint_state()
        if self.reward_mode == "trajectory":
            self._closest_path_target(self._ee_position(), update_estimate=True)
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
        target_pos, target_vel, _, _ = self._desired_target(ee_pos)
        error_vec = target_pos - ee_pos
        reference_velocity = damped_velocity_ik(
            self.q,
            error_vec,
            target_vel,
            max_joint_speed=self.config.max_joint_speed,
        )
        command_velocity = policy_velocity_command(
            action,
            self.config.max_joint_speed,
            self.config.action_velocity_scale,
        )
        policy_velocity = command_velocity.copy()
        desired_q, command_velocity = integrate_joint_velocity(
            self.q,
            command_velocity,
            self.config.dt,
        )
        command_delta = command_velocity - self.prev_command
        self._publish_position_command(desired_q, command_velocity)
        self._spin_for(self.config.dt)

        self.t += self.config.dt
        self.step_count += 1

        ee_after = self._ee_position()
        closest_target_pos, closest_target_vel, _, trajectory_error = self._closest_path_target(
            ee_after,
        )
        ee_velocity = numerical_jacobian(self.q) @ self.qd

        desired_target_pos, desired_target_vel, _, desired_path_distance = self._desired_target(
            ee_after,
            update_estimate=True,
        )

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
            velocity_along_trajectory = (
                float(np.dot(ee_velocity, closest_target_vel / tangent_norm))
                if tangent_norm > 1e-6
                else 0.0
            )
            velocity_reward = (
                0.8 * float(np.clip(velocity_toward_path, -0.3, 0.3))
                + 1.4 * float(np.clip(velocity_along_trajectory, -0.3, 0.3))
            )
            position_reward = -6.0 * error + 0.30 * float(np.exp(-45.0 * error))
        else:
            reward_target_pos = desired_target_pos
            reward_target_vel = desired_target_vel
            error = float(np.linalg.norm(desired_target_pos - ee_after))
            velocity_error = float(np.linalg.norm(ee_velocity - desired_target_vel))
            velocity_toward_path = 0.0
            velocity_along_trajectory = 0.0
            velocity_reward = -1.2 * velocity_error + 0.25 * float(np.exp(-8.0 * velocity_error))
            position_reward = -7.0 * error + 0.45 * float(np.exp(-35.0 * error))

        slow_speed_penalty, ee_speed, expected_speed, min_expected_speed = _speed_floor_penalty(
            ee_velocity,
            reward_target_vel,
            self.config.min_ee_speed_fraction,
            self.config.slow_speed_penalty_weight,
        )
        smoothness = float(np.linalg.norm(command_delta))
        smoothness_penalty_weight = (
            self.config.phase_two_smoothness_penalty_weight
            if self.reward_mode == "timed"
            else self.config.smoothness_penalty_weight
        )
        limit_penalty = joint_limit_cost(self.q)
        ee_direction = hand_z_axis(self.q)
        orientation_alignment = float(
            np.clip(np.dot(ee_direction, self.orientation_target_direction), -1.0, 1.0)
        )
        orientation_reward = self.config.orientation_reward_weight * orientation_alignment
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
        # timed mode: follow the next path point projected from the current EE position.
        reward = (
            position_reward
            + velocity_reward
            - 0.01 * float(np.dot(command_velocity, command_velocity))
            - smoothness_penalty_weight * smoothness
            - 6.0 * limit_penalty
            + orientation_reward
            - slow_speed_penalty
            - collision_penalty
        )

        self.prev_command = command_velocity.copy()
        terminated = self.config.terminate_on_collision and step_collision
        truncated = self.step_count >= self.config.horizon
        info = {
            "time": self.t,
            "ee_pos": ee_after,
            "target_pos": reward_target_pos,
            "reward_target_pos": reward_target_pos,
            "reward_target_vel": reward_target_vel,
            "closest_target_pos": closest_target_pos,
            "error": error,
            "desired_error": float(np.linalg.norm(desired_target_pos - ee_after)),
            "timed_error": float(np.linalg.norm(desired_target_pos - ee_after)),
            "trajectory_error": trajectory_error,
            "desired_path_distance": desired_path_distance,
            "velocity_toward_path": velocity_toward_path,
            "velocity_along_trajectory": velocity_along_trajectory,
            "velocity_reward": velocity_reward,
            "position_reward": position_reward,
            "ee_speed": ee_speed,
            "expected_speed": expected_speed,
            "min_expected_speed": min_expected_speed,
            "slow_speed_penalty": slow_speed_penalty,
            "orientation_alignment": orientation_alignment,
            "orientation_reward": orientation_reward,
            "ee_direction": ee_direction,
            "orientation_target_direction": self.orientation_target_direction,
            "ee_velocity": ee_velocity,
            "smoothness": smoothness,
            "command_delta_norm": smoothness,
            "smoothness_penalty_weight": smoothness_penalty_weight,
            "smoothness_penalty": smoothness_penalty_weight * smoothness,
            "policy_velocity": policy_velocity,
            "reference_velocity": reference_velocity,
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
