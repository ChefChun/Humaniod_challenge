import numpy as np

from .kinematics import PANDA_Q_MAX, PANDA_Q_MIN


def policy_velocity_command(
    action: np.ndarray,
    max_joint_speed: float,
    action_velocity_scale: float = 1.0,
) -> np.ndarray:
    """Convert the normalized SAC action into the commanded joint velocity."""
    clipped_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    command_velocity = action_velocity_scale * max_joint_speed * clipped_action
    return np.clip(command_velocity, -max_joint_speed, max_joint_speed)


def integrate_joint_velocity(
    q: np.ndarray,
    command_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a joint-velocity command into the desired joint position."""
    q = np.asarray(q, dtype=float)
    command_velocity = np.asarray(command_velocity, dtype=float)

    desired_q = q + dt * command_velocity
    clipped_q = np.clip(desired_q, PANDA_Q_MIN, PANDA_Q_MAX)
    limited_by_joint_bounds = ~np.isclose(clipped_q, desired_q)
    if np.any(limited_by_joint_bounds):
        # Keep the published velocity consistent with the joint-limit-clipped position.
        command_velocity = command_velocity.copy()
        command_velocity[limited_by_joint_bounds] = (clipped_q[limited_by_joint_bounds] - q[limited_by_joint_bounds]) / dt

    return clipped_q, command_velocity
