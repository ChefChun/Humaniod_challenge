from dataclasses import dataclass

import numpy as np


TRAJECTORY_KINDS = ("circle", "figure8", "horizontal8")
DEFAULT_TRAJECTORY_CENTER = (0.417393680355622, 0.0, 0.455758296155317)
DEFAULT_TRAJECTORY_RADIUS = 0.08
DEFAULT_TRAJECTORY_PERIOD = 6.0


# The target trajectory is a moving point in Cartesian space, not a joint-space path.
# Training and policy_runner both call target_at(t, cfg) to know where the robot
# end effector should be at the current time.
@dataclass(frozen=True)
class TrajectoryConfig:
    kind: str = "figure8"
    center: tuple[float, float, float] = DEFAULT_TRAJECTORY_CENTER
    radius: float = DEFAULT_TRAJECTORY_RADIUS
    period: float = DEFAULT_TRAJECTORY_PERIOD
    unreachable: bool = False


def make_trajectory_config(
    kind: str = "figure8",
    center: tuple[float, float, float] | list[float] = DEFAULT_TRAJECTORY_CENTER,
    radius: float = DEFAULT_TRAJECTORY_RADIUS,
    period: float = DEFAULT_TRAJECTORY_PERIOD,
    unreachable: bool = False,
) -> TrajectoryConfig:
    """Build and validate a trajectory config from CLI/JSON-friendly values."""
    center_values = tuple(float(value) for value in center)
    if len(center_values) != 3:
        raise ValueError(f"Trajectory center must have 3 values, got {len(center_values)}")
    if radius < 0.0:
        raise ValueError(f"Trajectory radius must be non-negative, got {radius}")
    if period <= 0.0:
        raise ValueError(f"Trajectory period must be positive, got {period}")
    return TrajectoryConfig(
        kind=kind,
        center=center_values,
        radius=float(radius),
        period=float(period),
        unreachable=bool(unreachable),
    )


def target_at(t: float, cfg: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return desired Cartesian position, velocity, and phase."""
    # phase advances from 0 to 2*pi over one period, then repeats.
    omega = 2.0 * np.pi / cfg.period
    phase = omega * t
    center = np.asarray(cfg.center, dtype=float)
    radius = cfg.radius

    if cfg.kind == "circle":
        # x stays fixed; y/z trace a circle around center.
        pos = center + np.array([0.0, radius * np.cos(phase), radius * np.sin(phase)])
        vel = np.array([0.0, -radius * omega * np.sin(phase), radius * omega * np.cos(phase)])
    elif cfg.kind == "figure8":
        # z oscillates twice as fast as y, producing a figure-eight in the y-z plane.
        pos = center + np.array([0.0, radius * np.sin(phase), 0.55 * radius * np.sin(2.0 * phase)])
        vel = np.array(
            [
                0.0,
                radius * omega * np.cos(phase),
                1.1 * radius * omega * np.cos(2.0 * phase),
            ]
        )
    elif cfg.kind in {"horizontal8", "vertical8"}:
        # Horizontal figure-eight in the Panda base x-y plane at constant center z.
        # This is the previous horizontal8 rotated 90 degrees around the z axis,
        # centered on the nominal Panda home end-effector position.
        x_amp = 3.0 * radius
        y_amp = 2.3 * radius
        pos = center + np.array([-y_amp * np.sin(2.0 * phase), x_amp * np.sin(phase), 0.0])
        vel = np.array(
            [
                -2.0 * y_amp * omega * np.cos(2.0 * phase),
                x_amp * omega * np.cos(phase),
                0.0,
            ]
        )
    else:
        raise ValueError(f"Unknown trajectory kind: {cfg.kind}")

    if cfg.unreachable:
        # Optional stress-test segment: briefly shift the target farther from the robot.
        cycle_fraction = (t % cfg.period) / cfg.period
        if 0.42 < cycle_fraction < 0.58:
            pos = pos + np.array([0.38, 0.0, 0.18])
            vel = vel.copy()

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
