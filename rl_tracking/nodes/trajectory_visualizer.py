import argparse

import numpy as np

from ..core.kinematics import PANDA_JOINT_NAMES, forward_kinematics
from ..core.trajectories import TrajectoryConfig, target_at


# Visualization data flow:
# target_at() produces the desired path and moving target marker.
# /isaac_joint_states provides the current robot joints, and forward_kinematics()
# converts those joints to the green end-effector marker.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the configured target trajectory as ROS2 visualization markers.")
    parser.add_argument("--trajectory", choices=["circle", "figure8", "horizontal8"], default="figure8")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--topic", default="/rl_tracking/trajectory_markers")
    parser.add_argument("--joint-states-topic", default="/isaac_joint_states")
    parser.add_argument("--center", nargs=3, type=float, default=(0.02, 0.47, 0.36), metavar=("X", "Y", "Z"))
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--period", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--unreachable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import rclpy
        from geometry_msgs.msg import Point
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as exc:
        raise SystemExit("ROS2 visualization packages are not available. Source your ROS2 workspace first.") from exc

    cfg = TrajectoryConfig(
        kind=args.trajectory,
        center=tuple(args.center),
        radius=args.radius,
        period=args.period,
        unreachable=args.unreachable,
    )

    class TrajectoryVisualizer(Node):
        def __init__(self) -> None:
            super().__init__("rl_trajectory_visualizer")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.publisher = self.create_publisher(MarkerArray, args.topic, qos)
            # Joint states are only needed for the actual end-effector marker.
            self.joint_state_subscription = self.create_subscription(
                JointState,
                args.joint_states_topic,
                self.on_joint_state,
                10,
            )
            self.start_time = self.get_clock().now()
            self.path_points = self._sample_path()
            self.ee_pos: np.ndarray | None = None
            self.timer = self.create_timer(1.0 / args.rate_hz, self.publish_markers)
            self.status_timer = self.create_timer(2.0, self.log_status)
            self.get_logger().info(
                f"Publishing {args.trajectory} trajectory markers on {args.topic} "
                f"in frame '{args.frame_id}' with {len(self.path_points)} path points."
            )
            self.get_logger().info(f"Subscribing to Franka joint states on {args.joint_states_topic}.")
            self.get_logger().info("Use RViz MarkerArray display, or an Isaac Sim marker/debug-draw subscriber, to see it.")

        def on_joint_state(self, msg: JointState) -> None:
            positions = dict(zip(msg.name, msg.position))
            if not all(name in positions for name in PANDA_JOINT_NAMES):
                return
            # Use the same FK approximation as training so RViz shows the controller's EE estimate.
            q = np.array([positions[name] for name in PANDA_JOINT_NAMES], dtype=float)
            self.ee_pos = forward_kinematics(q)

        def _sample_path(self) -> list[Point]:
            points = []
            for idx in range(args.samples + 1):
                t = cfg.period * idx / args.samples
                pos, _, _ = target_at(t, cfg)
                points.append(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
            return points

        def _base_marker(self, marker_id: int, marker_type: int) -> Marker:
            # RViz/Isaac identify markers by namespace and id; reusing ids updates old markers.
            marker = Marker()
            marker.header.frame_id = args.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rl_tracking_trajectory"
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.lifetime = Duration(seconds=1.0).to_msg()
            marker.pose.orientation.w = 1.0
            return marker

        def publish_markers(self) -> None:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            target_pos, _, _ = target_at(elapsed, cfg)

            # Blue: full desired target path.
            path = self._base_marker(0, Marker.LINE_STRIP)
            path.scale.x = 0.01
            path.color.r = 0.1
            path.color.g = 0.65
            path.color.b = 1.0
            path.color.a = 0.95
            path.points = self.path_points

            # Orange: current desired target point.
            target = self._base_marker(1, Marker.SPHERE)
            target.pose.position.x = float(target_pos[0])
            target.pose.position.y = float(target_pos[1])
            target.pose.position.z = float(target_pos[2])
            target.scale.x = 0.045
            target.scale.y = 0.045
            target.scale.z = 0.045
            target.color.r = 1.0
            target.color.g = 0.25
            target.color.b = 0.05
            target.color.a = 0.95

            markers = [path, target]
            if self.ee_pos is not None:
                # Green: current end-effector estimate from joint states.
                ee = self._base_marker(2, Marker.SPHERE)
                ee.pose.position.x = float(self.ee_pos[0])
                ee.pose.position.y = float(self.ee_pos[1])
                ee.pose.position.z = float(self.ee_pos[2])
                ee.scale.x = 0.04
                ee.scale.y = 0.04
                ee.scale.z = 0.04
                ee.color.r = 0.05
                ee.color.g = 0.9
                ee.color.b = 0.25
                ee.color.a = 0.95

                # Yellow: instantaneous tracking error between actual EE and target.
                error = self._base_marker(3, Marker.LINE_STRIP)
                error.scale.x = 0.006
                error.color.r = 1.0
                error.color.g = 1.0
                error.color.b = 0.0
                error.color.a = 0.85
                error.points = [
                    Point(x=float(self.ee_pos[0]), y=float(self.ee_pos[1]), z=float(self.ee_pos[2])),
                    Point(x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2])),
                ]
                markers.extend([ee, error])

            self.publisher.publish(MarkerArray(markers=markers))

        def log_status(self) -> None:
            subscribers = self.publisher.get_subscription_count()
            joint_publishers = self.count_publishers(args.joint_states_topic)
            if subscribers == 0:
                self.get_logger().warn(f"No subscribers on {args.topic}; nothing will be visible yet.")
            else:
                self.get_logger().info(f"{subscribers} subscriber(s) connected on {args.topic}.")
            if joint_publishers == 0:
                self.get_logger().warn(f"No publishers on {args.joint_states_topic}; actual end-effector marker is unavailable.")

    rclpy.init()
    node = TrajectoryVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
