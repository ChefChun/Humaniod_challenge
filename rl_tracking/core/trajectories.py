from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryConfig:
    kind: str = "figure8"
    center: tuple[float, float, float] = (0.02, 0.47, 0.36)
    radius: float = 0.08
    period: float = 6.0
    unreachable: bool = False


def target_at(t: float, cfg: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return desired Cartesian position, velocity, and phase."""
    omega = 2.0 * np.pi / cfg.period
    phase = omega * t
    center = np.asarray(cfg.center, dtype=float)
    radius = cfg.radius

    if cfg.kind == "circle":
        pos = center + np.array([0.0, radius * np.cos(phase), radius * np.sin(phase)])
        vel = np.array([0.0, -radius * omega * np.sin(phase), radius * omega * np.cos(phase)])
    elif cfg.kind == "figure8":
        pos = center + np.array([0.0, radius * np.sin(phase), 0.55 * radius * np.sin(2.0 * phase)])
        vel = np.array(
            [
                0.0,
                radius * omega * np.cos(phase),
                1.1 * radius * omega * np.cos(2.0 * phase),
            ]
        )
    else:
        raise ValueError(f"Unknown trajectory kind: {cfg.kind}")

    if cfg.unreachable:
        cycle_fraction = (t % cfg.period) / cfg.period
        if 0.42 < cycle_fraction < 0.58:
            pos = pos + np.array([0.38, 0.0, 0.18])
            vel = vel.copy()

    return pos, vel, phase
