import numpy as np

from .kinematics import PANDA_Q_MAX, PANDA_Q_MIN


def policy_acceleration_command(
    action: np.ndarray,
    max_joint_accel: float,
    action_scale: float,
) -> np.ndarray:
    """Convert the normalized SAC action into the commanded joint acceleration."""
    return np.clip(action_scale * max_joint_accel * np.clip(np.asarray(action, dtype=float), -1.0, 1.0), -max_joint_accel, max_joint_accel)


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

    # Jerk limiting prevents frame-to-frame acceleration jumps from reaching the robot.
    max_accel_delta = max_joint_jerk * dt
    acceleration_delta = np.clip(
        desired_acceleration - prev_acceleration,
        -max_accel_delta,
        max_accel_delta,
    )
    acceleration = prev_acceleration + acceleration_delta
    # First integration: acceleration -> joint velocity.
    command_velocity = np.clip(
        command_velocity + dt * acceleration,
        -max_joint_speed,
        max_joint_speed,
    )

    # Second integration: joint velocity -> desired joint position for the ROS2 command.
    desired_q = q + dt * command_velocity
    clipped_q = np.clip(desired_q, PANDA_Q_MIN, PANDA_Q_MAX)
    limited_by_joint_bounds = ~np.isclose(clipped_q, desired_q)
    if np.any(limited_by_joint_bounds):
        # Keep the published velocity consistent with the joint-limit-clipped position.
        command_velocity = command_velocity.copy()
        command_velocity[limited_by_joint_bounds] = (clipped_q[limited_by_joint_bounds] - q[limited_by_joint_bounds]) / dt

    return clipped_q, command_velocity, acceleration, acceleration_delta
