import torch
import torch.nn as nn
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
        nn.Linear(state_dim,64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim),
        nn.Tanh(),
        )
    def forward(self,state):
        return self.network(state)