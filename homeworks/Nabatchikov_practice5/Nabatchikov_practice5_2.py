import random

import numpy as np
import torch

from Nabatchikov_practice5.Nabatchikov_practice5_1 import DQN, Qfunction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed()


class DQNHard(DQN):
    def __init__(
        self, state_dim, action_dim, layer_size, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
        epsilon_min=0.01, target_update_freq=100
    ):
        super().__init__(state_dim, action_dim, layer_size, gamma, lr, batch_size, epsilon_decrease, epsilon_min)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim, layer_size)  # основная сеть
        self.target_q_function = Qfunction(self.state_dim, self.action_dim, layer_size)  # добавляем целевую сеть
        self.target_q_function.load_state_dict(self.q_function.state_dict())  # копируем параметры из основной сети
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        self.timestep = 0

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

            # Обновление целевой сети
            self.timestep += 1
            if self.timestep % self.target_update_freq == 0:
                self.target_q_function.load_state_dict(self.q_function.state_dict())


class DQNSoft(DQN):
    def __init__(
        self, state_dim, action_dim, layer_size, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
        epsilon_min=0.01, tau=0.01
    ):
        super().__init__(state_dim, action_dim, layer_size, gamma, lr, batch_size, epsilon_decrease, epsilon_min)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim, layer_size)  # основная сеть
        self.target_q_function = Qfunction(self.state_dim, self.action_dim, layer_size)  # добавляем целевую сеть
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)
        self.tau = tau

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

            # Обновление целевой сети (SOFT Target Networks)
            for target_param, param in zip(self.target_q_function.parameters(), self.q_function.parameters()):
                target_param.data = self.tau * param.data + (1.0 - self.tau) * target_param.data.detach()
