import random

import gym
import numpy as np
import time

from gym import Env

STATE_N = 500
ACTION_N = 6

random.seed(42)
np.random.seed(42)


def flatten_list(arr):
    res = []
    for obj in arr:
        res += flatten_list(obj) if isinstance(obj, (list, tuple, set)) else [obj]
    return res


class CrossEntropyAgent_V2:
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
        num_policy=10
    ):
        env = gym.make("Taxi-v3")
        mean_iteration_rewards = []
        for iteration in range(iteration_n):
            policies = {}
            # policy evaluation
            for i in range(num_policy):
                trajectories = [
                    self.get_trajectory(env=env, max_len=max_length)
                    for _ in range(trajectory_n)
                ]

                total_rewards = [
                    np.sum(trajectory["rewards"]) for trajectory in trajectories
                ]
                policies[i] = {"trajectories": trajectories, "total_rewards": total_rewards}
            reward_for_metric = flatten_list([p["total_rewards"] for p in policies.values()])
            all_policies_rewards = [np.sum(p["total_rewards"]) for p in policies.values()]
            info = {
                "iteration:": iteration,
                "mean_total_reward": np.mean(reward_for_metric),
                "max_total_reward": np.max(reward_for_metric),
            }
            if debug:
                print(info)
            mean_iteration_rewards.append(info)

            # policy improvement
            quantile = np.quantile(all_policies_rewards, q_param)
            elite_trajectories = []
            for policy, r in zip(policies.values(), all_policies_rewards):
                if r > quantile:
                    elite_trajectories.extend(policy["trajectories"])

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
    agent = CrossEntropyAgent_V2(state_n=STATE_N, action_n=ACTION_N)

    results = agent.train(
        trajectory_n=400,
        iteration_n=40,
        max_length=1000,
        q_param=0.6,
        laplace_smoothing=0.0,
        policy_smoothing=1.0,
        debug=True,
        num_policy = 10
    )

    print("Средняя награда на последней итерации:", results["info"][-1]["mean_total_reward"])
    print("Максимальная награда на последней итерации:", results["info"][-1]["max_total_reward"])
    print(results)
    trajectory = agent.get_trajectory(env=results["env"], max_len=500, visualize=True)

