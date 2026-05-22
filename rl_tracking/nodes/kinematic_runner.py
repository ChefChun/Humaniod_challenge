import argparse

import numpy as np

from ..core.kinematics import (
    PANDA_JOINT_NAMES,
    PANDA_Q_MAX,
    PANDA_Q_MIN,
    damped_velocity_ik,
    forward_kinematics,
)
from ..core.trajectories import (
    DEFAULT_TRAJECTORY_CENTER,
    DEFAULT_TRAJECTORY_PERIOD,
    DEFAULT_TRAJECTORY_RADIUS,
    TRAJECTORY_KINDS,
    closest_target_on_trajectory,
    make_trajectory_config,
    target_at,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send direct damped-IK commands to make the Franka end effector follow a target trajectory."
    )
    parser.add_argument("--controller-topic", default="/isaac_joint_commands")
    parser.add_argument("--joint-states-topic", default="/isaac_joint_states")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--trajectory", choices=TRAJECTORY_KINDS, default="figure8")
    parser.add_argument(
        "--trajectory-center",
        nargs=3,
        type=float,
        default=DEFAULT_TRAJECTORY_CENTER,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--trajectory-radius", type=float, default=DEFAULT_TRAJECTORY_RADIUS)
    parser.add_argument("--trajectory-period", type=float, default=DEFAULT_TRAJECTORY_PERIOD)
    parser.add_argument("--trajectory-unreachable", action="store_true")
    parser.add_argument("--max-joint-speed", type=float, default=0.8)
    parser.add_argument("--kp", type=float, default=3.5)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument(
        "--start-mode",
        choices=["nearest", "fixed"],
        default="nearest",
        help="nearest starts from the closest point on the path; fixed starts at trajectory phase zero.",
    )
    parser.add_argument(
        "--approach-duration",
        type=float,
        default=3.0,
        help="Seconds to hold the nearest path point before advancing along the trajectory.",
    )
    parser.add_argument("--projection-samples", type=int, default=720)
    parser.add_argument("--duration", type=float, help="Optional run time in seconds before shutting down.")
    parser.add_argument("--log-period", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise SystemExit("ROS2 Python packages are not available. Source your ROS2 workspace first.") from exc

    args = parse_args()
    trajectory_cfg = make_trajectory_config(
        kind=args.trajectory,
        center=tuple(args.trajectory_center),
        radius=args.trajectory_radius,
        period=args.trajectory_period,
        unreachable=args.trajectory_unreachable,
    )

    class KinematicRunner(Node):
        def __init__(self) -> None:
            super().__init__("rl_kinematic_trajectory_runner")
            self.publisher = self.create_publisher(JointState, args.controller_topic, 10)
            self.subscription = self.create_subscription(JointState, args.joint_states_topic, self.on_joint_state, 10)
            self.q: np.ndarray | None = None
            self.qd = np.zeros(7, dtype=float)
            self.start_time = self.get_clock().now()
            self.last_command_elapsed: float | None = None
            self.trajectory_time_offset = 0.0
            self.approach_target_pos: np.ndarray | None = None
            self.last_log_time = -float("inf")
            self.stop_requested = False
            self.timer = self.create_timer(args.dt, self.publish_command)
            self.get_logger().info(
                f"Following {args.trajectory} trajectory on {args.controller_topic}; "
                f"listening to {args.joint_states_topic}."
            )

        def on_joint_state(self, msg: JointState) -> None:
            positions = dict(zip(msg.name, msg.position))
            velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
            if not all(name in positions for name in PANDA_JOINT_NAMES):
                return
            self.q = np.array([positions[name] for name in PANDA_JOINT_NAMES], dtype=float)
            self.qd = np.array([velocities.get(name, 0.0) for name in PANDA_JOINT_NAMES], dtype=float)
            if self.approach_target_pos is None:
                self.initialize_trajectory_clock()

        def elapsed(self) -> float:
            return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        def initialize_trajectory_clock(self) -> None:
            assert self.q is not None
            if args.start_mode == "nearest":
                ee_pos = forward_kinematics(self.q)
                closest_pos, _, phase, distance = closest_target_on_trajectory(
                    ee_pos,
                    trajectory_cfg,
                    samples=args.projection_samples,
                )
                omega = 2.0 * np.pi / trajectory_cfg.period
                self.trajectory_time_offset = phase / omega
                self.approach_target_pos = closest_pos
                self.get_logger().info(
                    "Starting from nearest trajectory point: "
                    f"phase={phase:.3f} distance={distance:.4f}m "
                    f"approach_duration={args.approach_duration:.2f}s."
                )
            else:
                self.trajectory_time_offset = 0.0
                self.approach_target_pos = target_at(0.0, trajectory_cfg)[0]
                self.get_logger().info("Starting from fixed trajectory phase zero.")

        def current_target(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
            if elapsed < args.approach_duration and self.approach_target_pos is not None:
                return self.approach_target_pos, np.zeros(3, dtype=float)
            trajectory_time = max(0.0, elapsed - args.approach_duration) + self.trajectory_time_offset
            target_pos, target_vel, _ = target_at(trajectory_time, trajectory_cfg)
            return target_pos, target_vel

        def publish_command(self) -> None:
            if self.q is None:
                return

            elapsed = self.elapsed()
            if args.duration is not None and elapsed >= args.duration:
                self.publish_hold_command()
                self.get_logger().info("Requested duration reached; stopping kinematic runner.")
                self.stop_requested = True
                return

            target_pos, target_vel = self.current_target(elapsed)
            ee_pos = forward_kinematics(self.q)
            error_vec = target_pos - ee_pos
            command_velocity = damped_velocity_ik(
                self.q,
                error_vec,
                target_vel,
                kp=args.kp,
                damping=args.damping,
                max_joint_speed=args.max_joint_speed,
            )

            command_dt = args.dt
            if self.last_command_elapsed is not None:
                command_dt = float(np.clip(elapsed - self.last_command_elapsed, 1e-3, 0.1))
            self.last_command_elapsed = elapsed

            desired_q = self.q + command_dt * command_velocity
            clipped_q = np.clip(desired_q, PANDA_Q_MIN, PANDA_Q_MAX)
            limited = ~np.isclose(clipped_q, desired_q)
            if np.any(limited):
                command_velocity = command_velocity.copy()
                command_velocity[limited] = (clipped_q[limited] - self.q[limited]) / command_dt

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = PANDA_JOINT_NAMES
            msg.position = clipped_q.tolist()
            msg.velocity = command_velocity.tolist()
            self.publisher.publish(msg)

            if elapsed - self.last_log_time >= args.log_period:
                self.last_log_time = elapsed
                error = float(np.linalg.norm(error_vec))
                speed = float(np.linalg.norm(command_velocity))
                phase = "approach" if elapsed < args.approach_duration else "track"
                self.get_logger().info(
                    f"phase={phase} t={elapsed:.2f}s error={error:.4f}m joint_speed_norm={speed:.3f}"
                )

        def publish_hold_command(self) -> None:
            assert self.q is not None
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = PANDA_JOINT_NAMES
            msg.position = np.clip(self.q, PANDA_Q_MIN, PANDA_Q_MAX).tolist()
            msg.velocity = np.zeros(7, dtype=float).tolist()
            self.publisher.publish(msg)

    rclpy.init()
    node = KinematicRunner()
    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
