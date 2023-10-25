from typing import Any

import gym
import numpy as np
import torch
from gym import Env
from torch import nn

np.random.seed(42)


class CEM2(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.activation = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state) + torch.FloatTensor(np.random.uniform(-0.05, 0.05, 1))
        return self.activation(logits).detach().numpy()

    def get_trajectory(self, env: Env, trajectory_len=1000, visualize=False):
        trajectory = {'states': [], 'actions': [], 'total_reward': 0}

        state = env.reset()

        for _ in range(trajectory_len):
            trajectory['states'].append(state)
            action = self.get_action(state)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            trajectory['total_reward'] += reward

            if done:
                break

            if visualize:
                env.render()

        return trajectory

    @staticmethod
    def get_elite_trajectories(trajectories, q_param):
        total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
        quantile = np.quantile(total_rewards, q=q_param)
        return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states_tensor = torch.from_numpy(np.array(elite_states))
        elite_actions_tensor = torch.from_numpy(np.array(elite_actions))

        loss = self.loss(self.forward(elite_states_tensor), elite_actions_tensor)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self, env: Env, epochs: int, q_param: float, trajectory_n: int, trajectory_len: int, debug: bool = False):
        logs = []
        for epoch in range(epochs):
            trajectories = [self.get_trajectory(env=env, trajectory_len=trajectory_len) for _ in range(trajectory_n)]

            total_reward = [trajectory['total_reward'] for trajectory in trajectories]
            info = {
                "iteration:": epoch,
                "mean_total_reward": np.mean(total_reward),
                "max_total_reward": np.max(total_reward),
            }
            logs.append(info)
            if debug:
                print(info)

            elite_trajectories = self.get_elite_trajectories(trajectories=trajectories, q_param=q_param)

            if len(elite_trajectories) > 0:
                self.update_policy(elite_trajectories)
        return {
            "q_param": q_param,
            "trajectory_n": trajectory_n,
            "epochs": epochs,
            "trajectory_len": trajectory_len,
            "info": logs,
        }


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    agent = CEM2(state_dim=2)
    agent.fit(env=env, epochs=100, q_param=0.97, trajectory_n=1000, trajectory_len=999, debug=True)
    agent.get_trajectory(env=env, trajectory_len=1000, visualize=True)
