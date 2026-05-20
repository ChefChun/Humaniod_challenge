import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=self.device),
            "actions": torch.as_tensor(self.actions[indices], device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=self.device),
            "dones": torch.as_tensor(self.dones[indices], device=self.device),
        }


class TrackingEncoder(nn.Module):
    def __init__(self, features_dim: int = 128):
        super().__init__()
        self.robot_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.tracking_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.command_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(160, 192),
            nn.ReLU(),
            nn.Linear(192, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        robot_features = self.robot_encoder(obs[:, 0:14])
        tracking_features = self.tracking_encoder(obs[:, 14:28])
        command_features = self.command_encoder(obs[:, 28:35])
        return self.fusion(torch.cat([robot_features, tracking_features, command_features], dim=-1))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, features_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        if obs_dim != 35:
            raise ValueError(f"Expected obs_dim=35, got {obs_dim}")
        self.encoder = TrackingEncoder(features_dim)
        self.head = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.head(self.encoder(obs))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self(obs_t)
            action = torch.tanh(mean) if deterministic else self.sample(obs_t)[0]
        return action.squeeze(0).cpu().numpy()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, features_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        if obs_dim != 35:
            raise ValueError(f"Expected obs_dim=35, got {obs_dim}")
        self.encoder = TrackingEncoder(features_dim)
        self.q = nn.Sequential(
            nn.Linear(features_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.encoder(obs), action], dim=-1))


@dataclass
class SACConfig:
    obs_dim: int = 35
    action_dim: int = 7
    features_dim: int = 128
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float | None = None


class TorchSACAgent:
    def __init__(self, config: SACConfig, device: torch.device):
        self.config = config
        self.device = device
        self.actor = Actor(config.obs_dim, config.action_dim, config.features_dim, config.hidden_dim).to(device)
        self.critic1 = Critic(config.obs_dim, config.action_dim, config.features_dim, config.hidden_dim).to(device)
        self.critic2 = Critic(config.obs_dim, config.action_dim, config.features_dim, config.hidden_dim).to(device)
        self.target_critic1 = Critic(config.obs_dim, config.action_dim, config.features_dim, config.hidden_dim).to(device)
        self.target_critic2 = Critic(config.obs_dim, config.action_dim, config.features_dim, config.hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr)

        target_entropy = config.target_entropy
        if target_entropy is None:
            target_entropy = -float(config.action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(math.log(0.1), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.act(obs, self.device, deterministic=deterministic)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            backup = rewards + self.config.gamma * (1.0 - dones) * target_q

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(q1, backup)
        critic2_loss = F.mse_loss(q2, backup)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(obs)
        q_new = torch.min(self.critic1(obs, new_actions), self.critic2(obs, new_actions))
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_new.mean().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * src_param.data)

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "TorchSACAgent":
        payload = torch.load(path, map_location=device)
        agent = cls(SACConfig(**payload["config"]), device)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic1.load_state_dict(payload["critic1"])
        agent.critic2.load_state_dict(payload["critic2"])
        agent.target_critic1.load_state_dict(payload["target_critic1"])
        agent.target_critic2.load_state_dict(payload["target_critic2"])
        agent.log_alpha.data.copy_(payload["log_alpha"].to(device))
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic1_optimizer.load_state_dict(payload["critic1_optimizer"])
        agent.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        return agent
