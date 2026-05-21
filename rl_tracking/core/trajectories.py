from dataclasses import dataclass

import numpy as np


# The target trajectory is a moving point in Cartesian space, not a joint-space path.
# Training and policy_runner both call target_at(t, cfg) to know where the robot
# end effector should be at the current time.
@dataclass(frozen=True)
class TrajectoryConfig:
    kind: str = "figure8"
    center: tuple[float, float, float] = (0.02, 0.47, 0.36)
    radius: float = 0.08
    period: float = 6.0
    unreachable: bool = False


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
        # Large horizontal figure-eight above the Franka. With the default center,
        # x spans about -0.14-0.18 m, y spans about 0.35-0.59 m, and z stays at 0.52 m.
        pos = center + np.array([2.0 * radius * np.sin(phase), 1.5 * radius * np.sin(2.0 * phase), 0.16])
        vel = np.array(
            [
                2.0 * radius * omega * np.cos(phase),
                3.0 * radius * omega * np.cos(2.0 * phase),
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
    position = np.asarray(position, dtype=float)
    best_pos: np.ndarray | None = None
    best_vel: np.ndarray | None = None
    best_phase = 0.0
    best_distance = float("inf")

    for idx in range(samples):
        t = cfg.period * idx / samples
        pos, vel, phase = target_at(t, cfg)
        distance = float(np.linalg.norm(pos - position))
        if distance < best_distance:
            best_pos = pos
            best_vel = vel
            best_phase = phase
            best_distance = distance

    assert best_pos is not None
    assert best_vel is not None
    return best_pos, best_vel, best_phase, best_distance
