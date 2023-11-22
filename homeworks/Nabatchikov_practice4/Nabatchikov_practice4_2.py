import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def discretize_state(state, state_bins):
    p1, p2, p3, p4, p5, p6 = state
    theta1_bins, theta2_bins, sin_theta1_bins, sin_theta2_bins, theta1_dot_bins, theta2_dot_bins = state_bins
    theta1_bin = np.digitize(p1, theta1_bins)
    sin_theta1_bin = np.digitize(p2, sin_theta1_bins)
    theta2_bin = np.digitize(p3, theta2_bins)
    sin_theta2_bin = np.digitize(p4, sin_theta2_bins)
    theta1_dot_bin = np.digitize(p5, theta1_dot_bins)
    theta2_dot_bin = np.digitize(p6, theta2_dot_bins)
    return tuple([theta1_bin, sin_theta1_bin, theta2_bin, sin_theta2_bin, theta1_dot_bin, theta2_dot_bin])


def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    total_rewards = []
    action_n = 3
    num_states = 0
    n_finished = 0
    state_ranges = np.vstack((env.observation_space.low, env.observation_space.high)).T
    state_bins = [np.linspace(low, high, 20) for low, high in state_ranges]
    for bins in state_bins:
        num_states += len(bins)

    qfunction = dict()
    counter = dict()

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        state = discretize_state(state, state_bins)
        for i in range(trajectory_len):
            trajectory['states'].append(state)
            if state not in qfunction:
                qfunction[state] = np.zeros(action_n)
                counter[state] = np.zeros(action_n)


            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            state = discretize_state(state, state_bins)
            trajectory['rewards'].append(reward)

            if done:
                if i < trajectory_len - 1:
                    n_finished += 1
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
    print(f"n_finished: {n_finished}")
    return total_rewards


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)
    action_n = 3
    num_states = 0
    n_finished = 0
    state_ranges = np.vstack((env.observation_space.low, env.observation_space.high)).T
    state_bins = [np.linspace(low, high, 20) for low, high in state_ranges]
    for bins in state_bins:
        num_states += len(bins)

    qfunction = dict()

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        state = discretize_state(state, state_bins)
        if state not in qfunction:
            qfunction[state] = np.zeros(action_n)
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for i in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            if next_state not in qfunction:
                qfunction[next_state] = np.zeros(action_n)
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)
            if next_state not in qfunction:
                qfunction[next_state] = np.zeros(action_n)

            qfunction[state][action] += alpha * (reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])

            state = next_state
            action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            total_rewards[episode] += reward

            if done:
                if i < trajectory_len - 1:
                    n_finished += 1
                break
    print(f"n_finished: {n_finished}")
    return total_rewards


def q_learning(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)
    action_n = 3
    num_states = 0
    n_finished = 0
    state_ranges = np.vstack((env.observation_space.low, env.observation_space.high)).T
    state_bins = [np.linspace(low, high, 20) for low, high in state_ranges]
    for bins in state_bins:
        num_states += len(bins)

    qfunction = dict()

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        state = discretize_state(state, state_bins)
        if state not in qfunction:
            qfunction[state] = np.zeros(action_n)
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for i in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            if next_state not in qfunction:
                qfunction[next_state] = np.zeros(action_n)

            qfunction[state][action] += alpha * (
                reward + gamma * max(qfunction[next_state]) - qfunction[state][action]
            )

            state = next_state
            action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            total_rewards[episode] += reward

            if done:
                if i < trajectory_len - 1:
                    n_finished += 1
                break
    print(f"n_finished: {n_finished}")
    return total_rewards


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    total_rewards = SARSA(env, episode_n=1000, trajectory_len=1000, gamma=0.8)
    plt.plot(total_rewards)
    plt.show()
