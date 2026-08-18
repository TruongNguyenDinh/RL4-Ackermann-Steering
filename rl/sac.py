import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# Actor
# ==================================================

class SACActor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, state):

        x = self.network(state)

        mean = self.mean(x)

        log_std = self.log_std(x)

        # Giới hạn log std
        log_std = torch.clamp(
            log_std,
            -20,
            2
        )

        return mean, log_std

    def sample(self, state):

        mean, log_std = self.forward(state)

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std
        )

        z = distribution.rsample()

        action = torch.tanh(z)

        # Log probability sau tanh
        log_prob = distribution.log_prob(z)

        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True
        )

        mean_action = torch.tanh(mean)

        return (
            action,
            log_prob,
            mean_action
        )


# ==================================================
# Critic
# ==================================================

class SACCritic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
    ):
        super().__init__()

        input_dim = (
            state_dim
            + action_dim
        )

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state,
        action
    ):

        x = torch.cat(
            [state, action],
            dim=-1
        )

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2