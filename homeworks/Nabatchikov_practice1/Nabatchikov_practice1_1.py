import random

import gym
import numpy as np
import time

from gym import Env

STATE_N = 500
ACTION_N = 6

random.seed(42)
np.random.seed(42)


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.state_n = state_n
        self.action_n = action_n
        self.actions = np.arange(self.action_n)
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state: int) -> int:
        return int(np.random.choice(self.actions, p=self.model[state]))

    def get_trajectory(self, env: Env, max_len=1000, visualize=False):
        trajectory = {"states": [], "actions": [], "rewards": []}

        state = env.reset()

        for _ in range(max_len):
            trajectory["states"].append(state)

            action = self.get_action(state=state)
            trajectory["actions"].append(action)

            state, reward, done, _ = env.step(action=action)
            trajectory["rewards"].append(reward)

            if visualize:
                time.sleep(0.5)
                env.render()

            if done:
                break

        return trajectory

    def fit(
        self,
        elite_trajectories,
        laplace_smoothing: float = 0.0,
        policy_smoothing: float = 1.0,
    ) -> None:
        new_model = np.ones((self.state_n, self.action_n)) * laplace_smoothing
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_policy = new_model[state] / np.sum(new_model[state])
                old_policy = self.model[state].copy()
                new_model[state] = (
                    policy_smoothing * new_policy + (1 - policy_smoothing) * old_policy
                )
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None

    def train(
        self,
        trajectory_n=400,
        iteration_n=50,
        max_length=1000,
        q_param=0.6,
        laplace_smoothing=0.0,
        policy_smoothing=0.9,
        debug=False,
    ):
        env = gym.make("Taxi-v3")
        mean_iteration_rewards = []
        for iteration in range(iteration_n):
            # policy evaluation
            trajectories = [
                self.get_trajectory(env=env, max_len=max_length)
                for _ in range(trajectory_n)
            ]

            total_rewards = [
                np.sum(trajectory["rewards"]) for trajectory in trajectories
            ]
            info = {
                "iteration:": iteration,
                "mean_total_reward": np.mean(total_rewards),
                "max_total_reward": max(total_rewards),
            }
            if debug:
                print(info)
            mean_iteration_rewards.append(info)

            # policy improvement
            quantile = np.quantile(total_rewards, q_param)
            elite_trajectories = []
            for trajectory in trajectories:
                total_reward = np.sum(trajectory["rewards"])
                if total_reward > quantile:
                    elite_trajectories.append(trajectory)

            self.fit(
                elite_trajectories,
                laplace_smoothing=laplace_smoothing,
                policy_smoothing=policy_smoothing,
            )

        return {
            "q_param": q_param,
            "trajectory_n": trajectory_n,
            "iteration_n": iteration_n,
            "max_length": max_length,
            "info": mean_iteration_rewards,
            "env": env,
        }


if __name__ == "__main__":
    agent = CrossEntropyAgent(state_n=STATE_N, action_n=ACTION_N)

    results = agent.train(
        trajectory_n=400,
        iteration_n=40,
        max_length=1000,
        q_param=0.6,
        laplace_smoothing=0.0,
        policy_smoothing=1.0,
        debug=True
    )

    print("Средняя награда на последней итерации:", results["info"][-1]["mean_total_reward"])
    print("Максимальная награда на последней итерации:", results["info"][-1]["max_total_reward"])
    print(results)
    trajectory = agent.get_trajectory(env=results["env"], max_len=500, visualize=True)
