import numpy as np
import random
import torch
import torch.nn as nn


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim, layer_size=64):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, layer_size)
        self.linear_2 = nn.Linear(layer_size, layer_size)
        self.linear_3 = nn.Linear(layer_size, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions


class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        layer_size,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        epsilon_decrease=0.01,
        epsilon_min=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim, layer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(
                torch.tensor, list(zip(*batch))
            )

            targets = (
                rewards
                + self.gamma
                * (1 - dones)
                * torch.max(self.q_function(next_states), dim=1).values
            )
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

    def train(self, env, episode_n=100, t_max=500):
        rewards = []
        for episode in range(episode_n):
            total_reward = 0
            state = env.reset()
            for t in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                self.fit(state, action, reward, done, next_state)
                state = next_state
                if done:
                    break
            rewards.append(total_reward)
        return rewards, np.mean(rewards)
