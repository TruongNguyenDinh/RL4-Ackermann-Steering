import numpy as np
import torch
import torch.nn.functional as F

from rl.sac import (
    SACActor,
    SACCritic,
)


class SACAgent:

    def __init__(
        self,
        state_dim=22,
        action_dim=2,
        hidden_dim=256,
        device=None,
    ):

        # ==========================
        # Device
        # ==========================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # ==========================
        # Hyperparameters
        # ==========================

        self.gamma = 0.99
        self.tau = 0.005

        # ==========================
        # Actor
        # ==========================

        self.actor = SACActor(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(self.device)

        # ==========================
        # Critic
        # ==========================

        self.critic = SACCritic(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(self.device)

        # ==========================
        # Target Critic
        # ==========================

        self.target_critic = SACCritic(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(self.device)

        self.target_critic.load_state_dict(
            self.critic.state_dict()
        )

        # Target critic không train trực tiếp
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # ==========================
        # Optimizers
        # ==========================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=3e-4,
        )

        # ==========================
        # Alpha
        # ==========================

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=3e-4,
        )

        self.target_entropy = -float(
            action_dim
        )

    @property
    def alpha(self):

        return self.log_alpha.exp()

    # ==================================================
    # Select Action
    # ==================================================

    def select_action(
        self,
        state,
        evaluate=False,
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            if evaluate:

                mean, _ = self.actor(state)

                action = torch.tanh(mean)

            else:

                action, _, _ = (
                    self.actor.sample(state)
                )

        return action.cpu().numpy()[0]

    # ==================================================
    # UPDATE
    # ==================================================

    def update(
        self,
        replay_buffer,
        batch_size=256,
    ):

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay_buffer.sample(
            batch_size
        )

        # ==========================
        # Convert to Tensor
        # ==========================

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # ==================================================
        # 1. TARGET Q
        # ==================================================

        with torch.no_grad():

            next_actions, next_log_prob, _ = (
                self.actor.sample(next_states)
            )

            target_q1, target_q2 = (
                self.target_critic(
                    next_states,
                    next_actions,
                )
            )

            target_q = torch.min(
                target_q1,
                target_q2,
            )

            target_q = (
                target_q
                - self.alpha.detach()
                * next_log_prob
            )

            target = (
                rewards
                + (1.0 - dones)
                * self.gamma
                * target_q
            )

        # ==================================================
        # 2. CRITIC UPDATE
        # ==================================================

        current_q1, current_q2 = (
            self.critic(
                states,
                actions,
            )
        )

        critic_loss = (
            F.mse_loss(
                current_q1,
                target,
            )
            +
            F.mse_loss(
                current_q2,
                target,
            )
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # ==================================================
        # 3. ACTOR UPDATE
        # ==================================================

        new_actions, log_prob, _ = (
            self.actor.sample(states)
        )

        q1, q2 = self.critic(
            states,
            new_actions,
        )

        min_q = torch.min(
            q1,
            q2,
        )

        actor_loss = (
            self.alpha.detach()
            * log_prob
            - min_q
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # ==================================================
        # 4. ALPHA UPDATE
        # ==================================================

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob
                + self.target_entropy
            ).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # ==================================================
        # 5. SOFT UPDATE TARGET
        # ==================================================

        self.soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    # ==================================================
    # SOFT UPDATE
    # ==================================================

    def soft_update(self):

        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):

            target_param.data.copy_(
                self.tau * param.data
                + (1.0 - self.tau)
                * target_param.data
            )

    # ==================================================
    # SAVE
    # ==================================================

    def save(self, path):

        torch.save(
            {
                "actor":
                    self.actor.state_dict(),

                "critic":
                    self.critic.state_dict(),

                "target_critic":
                    self.target_critic.state_dict(),

                "actor_optimizer":
                    self.actor_optimizer.state_dict(),

                "critic_optimizer":
                    self.critic_optimizer.state_dict(),

                "log_alpha":
                    self.log_alpha.detach().cpu(),

                "alpha_optimizer":
                    self.alpha_optimizer.state_dict(),
            },
            path,
        )

    # ==================================================
    # LOAD
    # ==================================================

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.target_critic.load_state_dict(
            checkpoint["target_critic"]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        self.log_alpha.data.copy_(
            checkpoint["log_alpha"].to(
                self.device
            )
        )

        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer"]
        )