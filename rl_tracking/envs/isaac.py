import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.control import acceleration_residual_command, integrate_joint_acceleration
from ..core.kinematics import (
    PANDA_JOINT_NAMES,
    PANDA_Q_HOME,
    PANDA_Q_MAX,
    PANDA_Q_MIN,
    damped_velocity_ik,
    forward_kinematics,
    joint_limit_cost,
)
from ..core.trajectories import TrajectoryConfig, target_at


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
    residual_scale: float = 0.35
    max_joint_speed: float = 0.8
    max_joint_accel: float = 2.5
    max_joint_jerk: float = 18.0
    controller_topic: str = "/isaac_joint_commands"
    joint_states_topic: str = "/isaac_joint_states"
    reset_duration: float = 2.0
    command_duration: float = 0.12
    settle_timeout: float = 20.0
    seed: int = 7


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

        self._rclpy = rclpy
        self._JointState = JointState
        self._node = rclpy.create_node(f"isaac_franka_tracking_env_{id(self)}")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
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

        self.q: np.ndarray | None = None
        self.qd = np.zeros(7, dtype=float)
        self.command_velocity = np.zeros(7, dtype=float)
        self.prev_acceleration = np.zeros(7, dtype=float)
        self.step_count = 0
        self.t = 0.0
        self.trajectory_cfg = TrajectoryConfig(kind=config.trajectory)
        self._wait_for_joint_state()

    def _joint_state_cb(self, msg) -> None:
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if not all(name in positions for name in PANDA_JOINT_NAMES):
            return
        self.q = np.array([positions[name] for name in PANDA_JOINT_NAMES], dtype=float)
        self.qd = np.array([velocities.get(name, 0.0) for name in PANDA_JOINT_NAMES], dtype=float)

    def _spin_for(self, duration: float) -> None:
        deadline = time.time() + duration
        if self._node is not None:
            while time.time() < deadline:
                self._rclpy.spin_once(self._node, timeout_sec=0.01)

    def _wait_for_joint_state(self) -> None:
        start = time.time()
        if self._node is not None:
            while self.q is None:
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
            command = self._JointState()
            command.header.stamp = self._node.get_clock().now().to_msg()
            command.name = PANDA_JOINT_NAMES
            command.position = np.clip(q_desired, PANDA_Q_MIN, PANDA_Q_MAX).tolist()
            if qd_desired is not None:
                command.velocity = np.asarray(qd_desired, dtype=float).tolist()
            self._command_pub.publish(command)

    def _target(self) -> tuple[np.ndarray, np.ndarray, float]:
        return target_at(self.t, self.trajectory_cfg)

    def _ee_position(self) -> np.ndarray:
        assert self.q is not None
        return forward_kinematics(self.q)

    def _observe(self) -> np.ndarray:
        assert self.q is not None
        target_pos, target_vel, phase = self._target()
        obs = make_observation(
            self.q,
            self.qd,
            self.prev_acceleration,
            self._ee_position(),
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
        self.trajectory_cfg = TrajectoryConfig(kind=self.config.trajectory)

        random_offset = self.rng.normal(0.0, 0.025, size=7)
        reset_q = np.clip(PANDA_Q_HOME + random_offset, PANDA_Q_MIN, PANDA_Q_MAX)
        self._publish_position_command(reset_q, np.zeros(7))
        self._spin_for(self.config.reset_duration)
        self._wait_for_joint_state()
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.q is not None
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.config.action_noise > 0.0:
            action = np.clip(action + self.rng.normal(0.0, self.config.action_noise, size=7), -1.0, 1.0)

        target_pos, target_vel, _ = self._target()
        ee_pos = self._ee_position()
        error_vec = target_pos - ee_pos
        base_velocity = damped_velocity_ik(
            self.q,
            error_vec,
            target_vel,
            max_joint_speed=self.config.max_joint_speed,
        )
        desired_acceleration, base_acceleration, residual_acceleration = acceleration_residual_command(
            base_velocity,
            self.command_velocity,
            action,
            self.config.dt,
            self.config.max_joint_accel,
            self.config.residual_scale,
        )
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

        next_target_pos, _, _ = self._target()
        ee_after = self._ee_position()
        error = float(np.linalg.norm(next_target_pos - ee_after))
        smoothness = float(np.linalg.norm(acceleration_delta))
        jerk_norm = smoothness / self.config.dt
        limit_penalty = joint_limit_cost(self.q)

        reward = (
            -8.0 * error
            - 0.01 * float(np.dot(command_velocity, command_velocity))
            - 0.015 * float(np.dot(acceleration, acceleration))
            - 0.05 * smoothness
            - 6.0 * limit_penalty
            + 0.40 * float(np.exp(-35.0 * error))
        )

        self.prev_acceleration = acceleration.copy()
        terminated = False
        truncated = self.step_count >= self.config.horizon
        info = {
            "time": self.t,
            "ee_pos": ee_after,
            "target_pos": next_target_pos,
            "error": error,
            "smoothness": smoothness,
            "jerk_norm": jerk_norm,
            "base_velocity": base_velocity,
            "base_acceleration": base_acceleration,
            "residual_acceleration": residual_acceleration,
            "acceleration": acceleration,
            "command_velocity": command_velocity,
            "command": command_velocity,
            "is_success": error < 0.035,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
