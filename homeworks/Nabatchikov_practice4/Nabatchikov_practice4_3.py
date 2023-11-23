import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def MonteCarloEps(env, episode_n, trajectory_len=500, gamma=0.99, decay_rate=0.03):
    total_rewards = []
    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))
    counter = np.zeros((state_n, action_n))

    for episode in range(episode_n):
        epsilon = np.exp(-decay_rate * episode)

        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        for _ in range(trajectory_len):
            trajectory['states'].append(state)

            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)

            if done:
                break
        total_rewards.append(sum(trajectory['rewards']))

        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1

    return total_rewards


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    total_rewards = MonteCarlo(env, episode_n=1000, trajectory_len=1000, gamma=0.99)
    plt.plot(total_rewards)
    plt.show()
