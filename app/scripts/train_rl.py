# app/scripts/train_rl.py
import argparse
from typing import List

from app.rl_policy import RLTextFormatterAgent, Trajectory
from app.rl_env import run_single_episode


DEFAULT_TRAIN_PROMPTS = [
    "Explain reinforcement learning in one paragraph.",
    "What is the difference between Q-learning and policy gradients?",
    "Why might we use a discount factor in RL?",
    "Give a short explanation of the exploration-exploitation tradeoff.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--episodes-per-epoch", type=int, default=16)
    parser.add_argument("--model-path", type=str, default="rl_policy.pt")
    args = parser.parse_args()

    agent = RLTextFormatterAgent()

    for epoch in range(args.epochs):
        trajectories: List[Trajectory] = []
        total_reward = 0.0

        for i in range(args.episodes_per_epoch):
            prompt = DEFAULT_TRAIN_PROMPTS[i % len(DEFAULT_TRAIN_PROMPTS)]

            # 用当前策略 sample 一个模板动作
            action, _ = agent.select_action(prompt)
            episode = run_single_episode(prompt, action)

            traj = Trajectory(
                prompts=[prompt],
                actions=[action],
                rewards=[episode.reward],
                episode_results=[episode],
            )
            trajectories.append(traj)
            total_reward += episode.reward

        avg_reward = total_reward / args.episodes_per_epoch
        loss = agent.reinforce_update(trajectories)

        print(f"Epoch {epoch + 1}/{args.epochs} - avg_reward={avg_reward:.3f}, loss={loss:.4f}")

    # 保存策略
    agent.save(args.model_path)
    print(f"Saved RL policy to {args.model_path}")


if __name__ == "__main__":
    main()
