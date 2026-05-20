import numpy as np


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


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Approximate Franka Panda end-effector position from 7 joint angles."""
    q = np.asarray(q, dtype=float)
    a = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
    d = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107])
    alpha = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2])

    transform = np.eye(4)
    for i in range(7):
        transform = transform @ _dh(a[i], alpha[i], d[i], q[i])

    tool_offset = np.array([0.0, 0.0, 0.103, 1.0])
    return (transform @ tool_offset)[:3]


def numerical_jacobian(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Position-only finite-difference Jacobian, shape (3, 7)."""
    q = np.asarray(q, dtype=float)
    base = forward_kinematics(q)
    jac = np.zeros((3, 7), dtype=float)
    for idx in range(7):
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
    desired_cartesian_velocity = kp * position_error + target_velocity
    lhs = jac @ jac.T + (damping**2) * np.eye(3)
    qdot = jac.T @ np.linalg.solve(lhs, desired_cartesian_velocity)
    return np.clip(qdot, -max_joint_speed, max_joint_speed)


def joint_limit_cost(q: np.ndarray, margin: float = 0.12) -> float:
    """Soft cost that grows near joint limits."""
    span = PANDA_Q_MAX - PANDA_Q_MIN
    lower_dist = (q - PANDA_Q_MIN) / span
    upper_dist = (PANDA_Q_MAX - q) / span
    dist = np.minimum(lower_dist, upper_dist)
    return float(np.mean(np.maximum(0.0, margin - dist) ** 2))
