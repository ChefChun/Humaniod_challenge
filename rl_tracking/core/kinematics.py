import numpy as np


# This file is the geometry layer. It does not know about ROS2 or RL.
# It answers questions like:
# - Given 7 joint angles, where is the end effector?        forward_kinematics()
# - Given 7 joint angles, where is the hand link?           hand_position()
# - Given a target Cartesian velocity, how should joints move? damped_velocity_ik()
PANDA_BASE_FRAME = "panda_link0"
PANDA_EE_FRAME = "panda_hand"

PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

PANDA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
PANDA_Q_HOME = np.array([0.0, -0.45, 0.0, -2.2, 0.0, 1.75, 0.75])


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def _rotation_x(angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rotation_y(angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [ca, 0.0, sa, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sa, 0.0, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rotation_z(angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [ca, -sa, 0.0, 0.0],
            [sa, ca, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _panda_hand_transform(q: np.ndarray) -> np.ndarray:
    """Standard MoveIt/franka_description Panda base-to-hand transform."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"Expected 7 Panda joints, got shape {q.shape}")

    origins = [
        ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
        ((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        ((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        ((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        ((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ]

    transform = np.eye(4)
    for (xyz, rpy), joint_angle in zip(origins, q):
        transform = transform @ _translation(*xyz) @ _rpy(*rpy) @ _rotation_z(joint_angle)

    # Fixed panda_link8 and panda_hand transforms from franka_description.
    transform = transform @ _translation(0.0, 0.0, 0.107)
    transform = transform @ _translation(0.0, 0.0, 0.1034) @ _rpy(0.0, 0.0, -np.pi / 4.0)
    return transform


def hand_position(q: np.ndarray) -> np.ndarray:
    """Approximate Franka hand-link position from 7 joint angles."""
    return _panda_hand_transform(q)[:3, 3]


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Approximate Franka Panda end-effector position from 7 joint angles."""
    return hand_position(q)


def numerical_jacobian(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Position-only finite-difference Jacobian, shape (3, 7)."""
    q = np.asarray(q, dtype=float)
    base = forward_kinematics(q)
    jac = np.zeros((3, 7), dtype=float)
    for idx in range(7):
        # Move one joint slightly and measure how the end-effector position changes.
        shifted = q.copy()
        shifted[idx] += eps
        jac[:, idx] = (forward_kinematics(shifted) - base) / eps
    return jac


def damped_velocity_ik(
    q: np.ndarray,
    position_error: np.ndarray,
    target_velocity: np.ndarray,
    kp: float = 3.0,
    damping: float = 0.08,
    max_joint_speed: float = 1.4,
) -> np.ndarray:
    """Map desired Cartesian velocity to a smooth joint velocity command."""
    jac = numerical_jacobian(q)
    # Proportional correction pulls the end effector toward the target while target_velocity
    # makes it follow the moving path instead of only chasing the current point.
    desired_cartesian_velocity = kp * position_error + target_velocity
    # Damping makes the inverse stable near singular or awkward robot configurations.
    lhs = jac @ jac.T + (damping**2) * np.eye(3)
    qdot = jac.T @ np.linalg.solve(lhs, desired_cartesian_velocity)
    return np.clip(qdot, -max_joint_speed, max_joint_speed)


def joint_limit_cost(q: np.ndarray, margin: float = 0.12) -> float:
    """Soft cost that grows near joint limits."""
    # This is used in the RL reward to discourage learning motions near hard limits.
    span = PANDA_Q_MAX - PANDA_Q_MIN
    lower_dist = (q - PANDA_Q_MIN) / span
    upper_dist = (PANDA_Q_MAX - q) / span
    dist = np.minimum(lower_dist, upper_dist)
    return float(np.mean(np.maximum(0.0, margin - dist) ** 2))
