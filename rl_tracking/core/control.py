import numpy as np

from .kinematics import PANDA_Q_MAX, PANDA_Q_MIN


def acceleration_residual_command(
    base_velocity: np.ndarray,
    command_velocity: np.ndarray,
    action: np.ndarray,
    dt: float,
    max_joint_accel: float,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a normalized SAC action into a joint acceleration target."""
    base_acceleration = (np.asarray(base_velocity, dtype=float) - command_velocity) / dt
    base_acceleration = np.clip(base_acceleration, -max_joint_accel, max_joint_accel)
    residual_acceleration = residual_scale * max_joint_accel * np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    desired_acceleration = np.clip(
        base_acceleration + residual_acceleration,
        -max_joint_accel,
        max_joint_accel,
    )
    return desired_acceleration, base_acceleration, residual_acceleration


def integrate_joint_acceleration(
    q: np.ndarray,
    command_velocity: np.ndarray,
    prev_acceleration: np.ndarray,
    desired_acceleration: np.ndarray,
    dt: float,
    max_joint_speed: float,
    max_joint_accel: float,
    max_joint_jerk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply acceleration and jerk limits, then integrate to velocity and position."""
    q = np.asarray(q, dtype=float)
    command_velocity = np.asarray(command_velocity, dtype=float)
    prev_acceleration = np.asarray(prev_acceleration, dtype=float)
    desired_acceleration = np.clip(np.asarray(desired_acceleration, dtype=float), -max_joint_accel, max_joint_accel)

    max_accel_delta = max_joint_jerk * dt
    acceleration_delta = np.clip(
        desired_acceleration - prev_acceleration,
        -max_accel_delta,
        max_accel_delta,
    )
    acceleration = prev_acceleration + acceleration_delta
    command_velocity = np.clip(
        command_velocity + dt * acceleration,
        -max_joint_speed,
        max_joint_speed,
    )

    desired_q = q + dt * command_velocity
    clipped_q = np.clip(desired_q, PANDA_Q_MIN, PANDA_Q_MAX)
    limited_by_joint_bounds = ~np.isclose(clipped_q, desired_q)
    if np.any(limited_by_joint_bounds):
        command_velocity = command_velocity.copy()
        command_velocity[limited_by_joint_bounds] = (clipped_q[limited_by_joint_bounds] - q[limited_by_joint_bounds]) / dt

    return clipped_q, command_velocity, acceleration, acceleration_delta
