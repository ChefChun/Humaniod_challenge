import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import rclpy
import torch

from ..algorithms.sac import ReplayBuffer, SACConfig, TorchSACAgent
from ..envs.isaac import IsaacEnvConfig, IsaacFrankaTrackingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom PyTorch SAC tracker on Isaac Sim Franka.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--save-dir", default="runs/torch_isaac")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--trajectory", choices=["circle", "figure8", "horizontal8"], default="figure8")
    parser.add_argument("--controller-topic", default="/isaac_joint_commands")
    parser.add_argument("--joint-states-topic", default="/isaac_joint_states")
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--settle-timeout", type=float, default=20.0)
    parser.add_argument("--obs-noise", type=float, default=0.001)
    parser.add_argument("--action-noise", type=float, default=0.01)
    parser.add_argument("--max-joint-speed", type=float, default=0.8)
    parser.add_argument("--max-joint-accel", type=float, default=2.5)
    parser.add_argument("--max-joint-jerk", type=float, default=18.0)
    parser.add_argument("--action-accel-scale", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--curriculum-switch-min-episodes", type=int, default=5)
    parser.add_argument("--curriculum-switch-window", type=int, default=5)
    parser.add_argument("--curriculum-switch-trajectory-error", type=float, default=0.045)
    parser.add_argument("--trajectory-projection-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def make_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "training_metrics.csv"

    env_config = IsaacEnvConfig(
        dt=args.dt,
        horizon=args.horizon,
        trajectory=args.trajectory,
        obs_noise=args.obs_noise,
        action_noise=args.action_noise,
        action_accel_scale=args.action_accel_scale if args.residual_scale is None else args.residual_scale,
        max_joint_speed=args.max_joint_speed,
        max_joint_accel=args.max_joint_accel,
        max_joint_jerk=args.max_joint_jerk,
        trajectory_projection_samples=args.trajectory_projection_samples,
        controller_topic=args.controller_topic,
        joint_states_topic=args.joint_states_topic,
        settle_timeout=args.settle_timeout,
        seed=args.seed,
    )
    sac_config = SACConfig(
        obs_dim=35,
        action_dim=7,
        features_dim=args.features_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
    )
    (save_dir / "isaac_env_config.json").write_text(json.dumps(env_config.__dict__, indent=2) + "\n")
    (save_dir / "sac_config.json").write_text(json.dumps(sac_config.__dict__, indent=2) + "\n")

    device = make_device(args.device)
    replay = ReplayBuffer(sac_config.obs_dim, sac_config.action_dim, args.buffer_size, device)
    agent = TorchSACAgent(sac_config, device)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    except Exception:
        writer = None

    rclpy.init()
    env = None
    try:
        env = IsaacFrankaTrackingEnv(env_config)
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0
        episode_idx = 0

        with metrics_path.open("w", newline="") as csv_file:
            fieldnames = [
                "step",
                "episode",
                "episode_return",
                "episode_length",
                "reward",
                "tracking_error",
                "timed_error",
                "trajectory_error",
                "velocity_toward_path",
                "velocity_reward",
                "position_reward",
                "smoothness",
                "jerk_norm",
                "command_velocity_norm",
                "acceleration_norm",
                "actor_loss",
                "critic1_loss",
                "critic2_loss",
                "alpha",
            ]
            writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_csv.writeheader()

            print("Training custom PyTorch SAC on Isaac Sim Franka")
            print(f"save_dir: {save_dir}")
            print(f"device: {device}")
            print(f"controller_topic: {env_config.controller_topic}")
            print(f"joint_states_topic: {env_config.joint_states_topic}")
            print(
                "control: RL acceleration policy "
                f"max_speed={env_config.max_joint_speed} "
                f"max_accel={env_config.max_joint_accel} "
                f"max_jerk={env_config.max_joint_jerk} "
                f"action_scale={env_config.action_accel_scale}"
            )
            print(
                "reward curriculum: trajectory-path reward first, timed target reward after "
                f"{args.curriculum_switch_window} recent episode(s) average trajectory error "
                f"< {args.curriculum_switch_trajectory_error}"
            )

            last_losses: dict[str, float] = {}
            reward_mode = "trajectory"
            recent_episode_trajectory_errors: list[float] = []
            episode_trajectory_error_sum = 0.0
            for step in range(1, args.total_timesteps + 1):
                if step < args.learning_starts:
                    action = env.action_space.sample()
                else:
                    action = agent.act(obs, deterministic=False)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.add(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_return += reward
                episode_length += 1
                episode_trajectory_error_sum += float(info.get("trajectory_error", 0.0))

                if step >= args.update_after and replay.size >= args.batch_size and step % args.update_every == 0:
                    for _ in range(args.updates_per_step):
                        last_losses = agent.update(replay.sample(args.batch_size))

                if writer is not None:
                    writer.add_scalar("tracking/error_m", info.get("error", 0.0), step)
                    writer.add_scalar("tracking/timed_error_m", info.get("timed_error", 0.0), step)
                    writer.add_scalar("tracking/trajectory_error_m", info.get("trajectory_error", 0.0), step)
                    writer.add_scalar("tracking/velocity_toward_path", info.get("velocity_toward_path", 0.0), step)
                    writer.add_scalar("reward/velocity", info.get("velocity_reward", 0.0), step)
                    writer.add_scalar("reward/position", info.get("position_reward", 0.0), step)
                    writer.add_scalar("tracking/smoothness", info.get("smoothness", 0.0), step)
                    writer.add_scalar("tracking/jerk_norm", info.get("jerk_norm", 0.0), step)
                    # Control norms reveal saturation even when reward still appears to improve.
                    writer.add_scalar(
                        "control/command_velocity_norm",
                        float(np.linalg.norm(info.get("command_velocity", np.zeros(7)))),
                        step,
                    )
                    writer.add_scalar(
                        "control/acceleration_norm",
                        float(np.linalg.norm(info.get("acceleration", np.zeros(7)))),
                        step,
                    )
                    writer.add_scalar("train/reward", reward, step)
                    for key, value in last_losses.items():
                        writer.add_scalar(f"loss/{key}", value, step)

                if step % args.log_freq == 0:
                    row = {
                        "step": step,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "reward": reward,
                        "tracking_error": info.get("error", 0.0),
                        "timed_error": info.get("timed_error", 0.0),
                        "trajectory_error": info.get("trajectory_error", 0.0),
                        "velocity_toward_path": info.get("velocity_toward_path", 0.0),
                        "velocity_reward": info.get("velocity_reward", 0.0),
                        "position_reward": info.get("position_reward", 0.0),
                        "smoothness": info.get("smoothness", 0.0),
                        "jerk_norm": info.get("jerk_norm", 0.0),
                        "command_velocity_norm": float(np.linalg.norm(info.get("command_velocity", np.zeros(7)))),
                        "acceleration_norm": float(np.linalg.norm(info.get("acceleration", np.zeros(7)))),
                        "actor_loss": last_losses.get("actor_loss", 0.0),
                        "critic1_loss": last_losses.get("critic1_loss", 0.0),
                        "critic2_loss": last_losses.get("critic2_loss", 0.0),
                        "alpha": last_losses.get("alpha", 0.0),
                    }
                    writer_csv.writerow(row)
                    csv_file.flush()
                    print(
                        f"step={step} ep={episode_idx} "
                        f"reward={reward:.3f} err={row['tracking_error']:.4f} "
                        f"smooth={row['smoothness']:.4f} return={episode_return:.2f}"
                    )

                if step % args.save_freq == 0:
                    agent.save(
                        str(save_dir / f"checkpoint_{step}.pt"),
                        extra={"step": step, "episode": episode_idx, "env_config": env_config.__dict__},
                    )

                if done:
                    if writer is not None:
                        writer.add_scalar("episode/return", episode_return, episode_idx)
                        writer.add_scalar("episode/length", episode_length, episode_idx)
                    if episode_length > 0:
                        mean_trajectory_error = episode_trajectory_error_sum / episode_length
                        recent_episode_trajectory_errors.append(mean_trajectory_error)
                        recent_episode_trajectory_errors = recent_episode_trajectory_errors[-args.curriculum_switch_window :]
                        can_switch = (
                            reward_mode == "trajectory"
                            and episode_idx + 1 >= args.curriculum_switch_min_episodes
                            and len(recent_episode_trajectory_errors) == args.curriculum_switch_window
                            and float(np.mean(recent_episode_trajectory_errors)) < args.curriculum_switch_trajectory_error
                        )
                        if can_switch:
                            reward_mode = "timed"
                            env.set_reward_mode("timed")
                            print(
                                "Curriculum switched to timed target reward "
                                f"after episode {episode_idx}; mean trajectory error "
                                f"{float(np.mean(recent_episode_trajectory_errors)):.4f}"
                            )
                    episode_idx += 1
                    obs, _ = env.reset(seed=args.seed + episode_idx)
                    episode_return = 0.0
                    episode_length = 0
                    episode_trajectory_error_sum = 0.0

        agent.save(
            str(save_dir / "final_model.pt"),
            extra={"step": args.total_timesteps, "episode": episode_idx, "env_config": env_config.__dict__},
        )
        print(f"Training completed. Final model saved to {save_dir / 'final_model.pt'}")
    finally:
        if env is not None:
            env.close()
        if writer is not None:
            writer.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
