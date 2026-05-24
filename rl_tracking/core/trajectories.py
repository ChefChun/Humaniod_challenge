from dataclasses import dataclass

import numpy as np


TRAJECTORY_KIND = "horizontal8"
DEFAULT_TRAJECTORY_CENTER = (0.417393680355622, 0.0, 0.455758296155317)
DEFAULT_TRAJECTORY_RADIUS = 0.08
DEFAULT_TRAJECTORY_PERIOD = 6.0


# The target trajectory is a moving point in Cartesian space, not a joint-space path.
# Training and policy_runner both call target_at(t, cfg) to know where the robot
# end effector should be at the current time.
@dataclass(frozen=True)
class TrajectoryConfig:
    center: tuple[float, float, float] = DEFAULT_TRAJECTORY_CENTER
    radius: float = DEFAULT_TRAJECTORY_RADIUS
    period: float = DEFAULT_TRAJECTORY_PERIOD


def make_trajectory_config() -> TrajectoryConfig:
    """Return the fixed horizontal figure-eight trajectory used everywhere."""
    return TrajectoryConfig()


def target_at(t: float, cfg: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return desired Cartesian position, velocity, and phase."""
    # phase advances from 0 to 2*pi over one period, then repeats.
    omega = 2.0 * np.pi / cfg.period
    phase = omega * t
    center = np.asarray(cfg.center, dtype=float)
    radius = cfg.radius

    # Horizontal figure-eight in the Panda base x-y plane at constant center z,
    # centered on the nominal Panda home end-effector position.
    x_amp = 2.2 * radius
    y_amp = 1.6 * radius
    pos = center + np.array([-y_amp * np.sin(2.0 * phase), x_amp * np.sin(phase), 0.0])
    vel = np.array(
        [
            -2.0 * y_amp * omega * np.cos(2.0 * phase),
            x_amp * omega * np.cos(phase),
            0.0,
        ]
    )

    return pos, vel, phase


def closest_target_on_trajectory(
    position: np.ndarray,
    cfg: TrajectoryConfig,
    samples: int = 180,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the sampled trajectory point closest to a Cartesian position."""
    pos, vel, phase, distance, _ = closest_target_on_trajectory_time(position, cfg, samples=samples)
    return pos, vel, phase, distance


def closest_target_on_trajectory_time(
    position: np.ndarray,
    cfg: TrajectoryConfig,
    samples: int = 180,
    center_time: float | None = None,
    search_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return the closest sampled trajectory point and its unwrapped trajectory time."""
    position = np.asarray(position, dtype=float)
    best_pos: np.ndarray | None = None
    best_vel: np.ndarray | None = None
    best_phase = 0.0
    best_time = 0.0
    best_distance = float("inf")

    if center_time is None or search_window is None or search_window >= cfg.period:
        times = (cfg.period * idx / samples for idx in range(samples))
    else:
        half_window = 0.5 * search_window
        times = (center_time - half_window + search_window * idx / max(1, samples - 1) for idx in range(samples))

    for t in times:
        pos, vel, phase = target_at(t, cfg)
        distance = float(np.linalg.norm(pos - position))
        if distance < best_distance:
            best_pos = pos
            best_vel = vel
            best_phase = phase
            best_time = t
            best_distance = distance

    assert best_pos is not None
    assert best_vel is not None
    return best_pos, best_vel, best_phase, best_distance, best_time
