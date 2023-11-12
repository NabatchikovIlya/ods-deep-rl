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


def init_policy(env):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def init_v_values(env):
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def policy_evaluation_step(env, v_values, policy, gamma):
    q_values = get_q_values(env, v_values, gamma)
    for state in env.get_all_states():
        for action in env.get_possible_actions(state):
            v_values[state] += policy[state][action] * q_values[state][action]
    return v_values


def policy_evaluation(env, policy, gamma, eval_iter_n):
    v_values = init_v_values(env=env)
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(env, v_values, policy, gamma)
    q_values = get_q_values(env, v_values, gamma)
    return q_values


def policy_improvement(env, q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float("-inf")
        for action in env.get_possible_actions(state):
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy


def fit(gamma, iter_n=20, eval_iter_n=100):
    env = FrozenLakeEnv()
    policy = init_policy(env=env)
    for _ in range(iter_n):
        q_values = policy_evaluation(
            env=env, policy=policy, gamma=gamma, eval_iter_n=eval_iter_n
        )
        policy = policy_improvement(env, q_values)

    total_rewards = []

    for _ in range(1000):
        total_reward = 0
        state = env.reset()
        for _ in range(1000):
            action = np.random.choice(
                env.get_possible_actions(state), p=list(policy[state].values())
            )
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    return {"gamma": gamma, "mean_total_rewards": np.mean(total_rewards)}


def main(bounds):
    results = []
    for gamma in bounds:
        result = fit(gamma=gamma)
        results.append(result)
        print(result)
    return results


if __name__ == "__main__":
    main(np.arange(0.01, 0.10, 0.005))
