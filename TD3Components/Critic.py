import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, actor_action_dim: int, actor_state_dim: int, privilege_state_dim: int):
        super().__init__()
        input_dim = (actor_action_dim +actor_state_dim+ privilege_state_dim)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, action, actor_state, privileged_state):
        x = torch.cat(
            [action, actor_state, privileged_state],
            dim=-1
        )

        return self.q1(x), self.q2(x)

    def q1_forward(self, action, actor_state, privileged_state):
        x = torch.cat(
            [action, actor_state, privileged_state],
            dim=-1
        )

        return self.q1(x)