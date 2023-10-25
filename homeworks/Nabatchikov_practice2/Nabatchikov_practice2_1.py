import gym
import numpy as np
import torch
from gym import Env
from torch import nn

np.random.seed(42)


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_n)
        )

        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action

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
        elite_states = torch.from_numpy(np.array(elite_states))
        elite_actions = torch.from_numpy(np.array(elite_actions))

        loss = self.loss(self.forward(elite_states), elite_actions)
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
    env = gym.make('Acrobot-v1')
    agent = CEM(state_dim=6, action_n=3)
    agent.fit(env=env, epochs=40, q_param=0.6, trajectory_n=200, trajectory_len=600, debug=True)
    agent.get_trajectory(env=env, trajectory_len=1000, visualize=True)
