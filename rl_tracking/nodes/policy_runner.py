import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..algorithms.sac import TorchSACAgent
from ..core.control import acceleration_residual_command, integrate_joint_acceleration
from ..core.kinematics import PANDA_JOINT_NAMES, damped_velocity_ik, forward_kinematics
from ..core.trajectories import TrajectoryConfig, target_at
from ..envs.isaac import make_observation


# Runtime data flow after training:
# Isaac publishes joint states -> this node builds the same observation used during training ->
# the saved SAC actor predicts residual acceleration -> control.py integrates it into desired
# joint position/velocity -> this node publishes JointState commands back to Isaac.
def load_env_config(path: Path | None, model_path: Path) -> dict:
    if path is None:
        for candidate in [
            model_path.parent / "isaac_env_config.json",
            model_path.parent.parent / "isaac_env_config.json",
        ]:
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return {"trajectory": "figure8"}
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a learned PyTorch SAC tracker to Isaac Sim Franka topics.")
    parser.add_argument("--model", type=Path, default=Path("runs/torch_isaac/final_model.pt"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--controller-topic", default="/isaac_joint_commands")
    parser.add_argument("--joint-states-topic", default="/isaac_joint_states")
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--trajectory", choices=["circle", "figure8", "vertical8"])
    parser.add_argument("--max-joint-speed", type=float)
    parser.add_argument("--max-joint-accel", type=float)
    parser.add_argument("--max-joint-jerk", type=float)
    parser.add_argument("--residual-scale", type=float)
    return parser.parse_args()


class TorchPolicyAdapter:
    def __init__(self, path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The checkpoint contains the trained actor, critics, entropy temperature, and config.
        self.agent = TorchSACAgent.load(str(path), self.device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Deployment uses deterministic actions so the robot repeats the learned behavior.
        return self.agent.act(obs, deterministic=True)


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise SystemExit("ROS2 Python packages are not available. Source your ROS2 workspace first.") from exc

    args = parse_args()
    policy = TorchPolicyAdapter(args.model)
    env_config = load_env_config(args.config, args.model)
    trajectory_kind = args.trajectory or env_config.get("trajectory", "figure8")
    max_joint_speed = args.max_joint_speed if args.max_joint_speed is not None else float(env_config.get("max_joint_speed", 0.8))
    max_joint_accel = args.max_joint_accel if args.max_joint_accel is not None else float(env_config.get("max_joint_accel", 2.5))
    max_joint_jerk = args.max_joint_jerk if args.max_joint_jerk is not None else float(env_config.get("max_joint_jerk", 18.0))
    residual_scale = args.residual_scale if args.residual_scale is not None else float(env_config.get("residual_scale", 0.35))

    class TrackerNode(Node):
        def __init__(self) -> None:
            super().__init__("rl_ee_tracker")
            self.publisher = self.create_publisher(JointState, args.controller_topic, 10)
            self.subscription = self.create_subscription(JointState, args.joint_states_topic, self.on_joint_state, 10)
            self.q: np.ndarray | None = None
            self.qd = np.zeros(7)
            self.command_velocity = np.zeros(7)
            self.prev_acceleration = np.zeros(7)
            self.initialized = False
            self.start_time = self.get_clock().now()

        def on_joint_state(self, msg: JointState) -> None:
            # Every incoming Isaac joint-state message triggers one policy inference and command.
            positions = dict(zip(msg.name, msg.position))
            velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
            if not all(name in positions for name in PANDA_JOINT_NAMES):
                return

            self.q = np.array([positions[name] for name in PANDA_JOINT_NAMES], dtype=float)
            self.qd = np.array([velocities.get(name, 0.0) for name in PANDA_JOINT_NAMES], dtype=float)
            if not self.initialized:
                # Start integration from measured velocity to avoid a command discontinuity on first state.
                self.command_velocity = np.clip(self.qd, -max_joint_speed, max_joint_speed)
                self.initialized = True
            self.publish_command()

        def publish_command(self) -> None:
            assert self.q is not None
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            target_pos, target_vel, phase = target_at(elapsed, TrajectoryConfig(kind=trajectory_kind))
            ee_pos = forward_kinematics(self.q)
            # Reuse the exact observation builder from training so the policy sees familiar inputs.
            obs = make_observation(
                self.q,
                self.qd,
                self.prev_acceleration,
                ee_pos,
                target_pos,
                target_vel,
                phase,
                prev_action_scale=max_joint_accel,
            ).astype(np.float32)
            residual = policy.predict(obs)
            # Deployment mirrors IsaacFrankaTrackingEnv.step: residual action means acceleration.
            base_velocity = damped_velocity_ik(
                self.q,
                target_pos - ee_pos,
                target_vel,
                max_joint_speed=max_joint_speed,
            )
            desired_acceleration, _, _ = acceleration_residual_command(
                base_velocity,
                self.command_velocity,
                residual,
                args.dt,
                max_joint_accel,
                residual_scale,
            )
            desired_q, command_velocity, acceleration, _ = integrate_joint_acceleration(
                self.q,
                self.command_velocity,
                self.prev_acceleration,
                desired_acceleration,
                args.dt,
                max_joint_speed,
                max_joint_accel,
                max_joint_jerk,
            )
            self.command_velocity = command_velocity
            self.prev_acceleration = acceleration

            # ROS2 message boundary: Isaac receives these arrays by joint name.
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = PANDA_JOINT_NAMES
            msg.position = desired_q.tolist()
            msg.velocity = command_velocity.tolist()
            self.publisher.publish(msg)

    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
