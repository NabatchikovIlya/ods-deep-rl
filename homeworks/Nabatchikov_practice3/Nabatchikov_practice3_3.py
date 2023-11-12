import numpy as np

from Frozen_Lake import FrozenLakeEnv

np.random.seed(42)


def get_q_values(env, v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(
                    state, action, next_state
                ) * env.get_reward(state, action, next_state)
                q_values[state][action] += (
                    gamma
                    * env.get_transition_prob(state, action, next_state)
                    * v_values[next_state]
                )
    return q_values


def init_v_values(env):
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def value_iteration_step(env, v_values, gamma):
    new_v_values = init_v_values(env)
    q_values = get_q_values(env, v_values, gamma)
    for state in env.get_all_states():
        if env.get_possible_actions(state):
            max_q_value = max(q_values[state].values())
            new_v_values[state] = max_q_value
    return new_v_values


def value_iteration(env, gamma, iter_n=100):
    v_values = init_v_values(env)
    for _ in range(iter_n):
        v_values = value_iteration_step(env, v_values, gamma)
    return v_values


def fit(gamma, iter_n=20):
    env = FrozenLakeEnv()

    v_values = value_iteration(env, gamma, iter_n)
    total_rewards = []
    total_env_infer_count = []

    for _ in range(100):
        total_reward = 0
        infer_count = 0
        state = env.reset()
        for step in range(1000):
            index_action = np.argmax(
                [
                    get_q_values(env, v_values, gamma)[state][a]
                    for a in env.get_possible_actions(state)
                ]
            )
            action = env.get_possible_actions(state)[index_action]
            state, reward, done, _ = env.step(action)
            infer_count += 1
            total_reward += reward
            if done:
                break

        total_rewards.append(total_reward)
        total_env_infer_count.append(infer_count)

    return {
        "gamma": gamma,
        "mean_total_rewards": np.mean(total_rewards),
        "env_infer_count": np.mean(total_env_infer_count),
    }


def main(bounds):
    results = []
    for gamma in bounds:
        result = fit(gamma=gamma)
        results.append(result)
        print(result)
    return results


if __name__ == "__main__":
    main(np.arange(0.03, 1, 0.03))
