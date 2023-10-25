import matplotlib.pyplot as plt
import torch
from torch import nn

data_n = 500
x_data = torch.linspace(-5, 5, data_n)
nu, sigma = torch.tensor(0.2), torch.tensor(0.2)
noise = torch.tensor([torch.normal(nu, sigma) for _ in range(len(x_data))])
y_data = torch.sin(x_data) + noise


class Network(nn.Module):
    def __init__(self, episode_n):
        super().__init__()
        self.episode_n = episode_n
        self.linear1 = nn.Linear(1, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        
    def forward(self, _input):
        hidden = self.linear1(_input)
        hidden = self.relu(hidden)
        output = self.linear2(hidden)
        return output
    
    def learn(self, x_data, y_data):
        for episode in range(self.episode_n):
            y_pred = self.forward(x_data)
            loss = torch.mean((y_data - y_pred) ** 2)
            print(f'episode: {episode}, loss = {loss}')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


network = Network(1000)
network.learn(x_data.reshape(data_n,1), y_data.reshape(data_n,1))
y_pred = network(x_data.reshape(data_n,1))

plt.scatter(x_data, y_data)
plt.plot(x_data.numpy(), y_pred.data.numpy(), 'r') 
plt.show()
