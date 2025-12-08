# app/rl_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .rl_env import TEMPLATES, run_single_episode, EpisodeResult


def prompt_to_vector(prompt: str, max_len: int = 64) -> torch.Tensor:
    """
    非常简单粗暴的 prompt 向量化方式：
    取前 max_len 个字符的 ASCII code / 255 做归一化。
    （作业重点是 RL 流程，不是 NLP embedding 精度，所以这样够用了）
    """
    arr = [ord(c) / 255.0 for c in prompt[:max_len]]
    if len(arr) < max_len:
        arr += [0.0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.float32)


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, num_actions: int | None = None):
        super().__init__()
        if num_actions is None:
            num_actions = len(TEMPLATES)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]
        return self.net(x)


@dataclass
class Trajectory:
    prompts: List[str]
    actions: List[int]
    rewards: List[float]
    episode_results: List[EpisodeResult]


class RLTextFormatterAgent:
    """
    使用 REINFORCE 的策略网络：
    输入 prompt 的简单向量，输出要选哪个模板。
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 1.0,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, prompt: str) -> Tuple[int, torch.Tensor]:
        """
        给定 prompt，sample 一个模板 index，并返回 log_prob（用于 REINFORCE）
        """
        x = prompt_to_vector(prompt).to(self.device)
        logits = self.policy(x.unsqueeze(0))  # [1, num_actions]
        probs = torch.softmax(logits, dim=-1)  # [1, num_actions]

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), log_prob

    def generate_with_policy(self, prompt: str) -> EpisodeResult:
        """
        用当前策略选择模板并生成一次回答（不更新参数，用于在线服务）
        """
        action, _ = self.select_action(prompt)
        episode = run_single_episode(prompt, action)
        return episode

    def reinforce_update(self, trajectories: List[Trajectory]) -> float:
        """
        使用 REINFORCE:
        J(θ) 的无偏梯度估计:  ∑ G_t * log π(a_t | s_t)
        这里我们简单一点：每个 traj 就一个 step（一个 prompt -> 一个 action）。
        """
        all_loss_terms = []

        for traj in trajectories:
            for prompt, action, reward in zip(traj.prompts, traj.actions, traj.rewards):
                x = prompt_to_vector(prompt).to(self.device)
                logits = self.policy(x.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                log_prob = dist.log_prob(torch.tensor(action, device=self.device))
                all_loss_terms.append(-log_prob * reward)  # maximize reward -> minimize -log_prob * reward

        if not all_loss_terms:
            return 0.0

        loss = torch.stack(all_loss_terms).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)


# ===== 全局单例，FastAPI/脚本都用同一个 agent =====
_agent_instance: RLTextFormatterAgent | None = None


def get_agent() -> RLTextFormatterAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RLTextFormatterAgent()
    return _agent_instance
