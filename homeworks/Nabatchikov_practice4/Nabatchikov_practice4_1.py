import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def q_learning(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)

    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for _ in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)

            qfunction[state][action] += alpha * (
                reward + gamma * max(qfunction[next_state]) - qfunction[state][action]
            )

            state = next_state
            action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            total_rewards[episode] += reward

            if done:
                break

    return total_rewards


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    total_rewards = q_learning(env, episode_n=1000, trajectory_len=1000, gamma=0.99)
    plt.plot(total_rewards)
    plt.show()
