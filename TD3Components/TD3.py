import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from TD3Components.Actor import Actor
from TD3Components.Critic import Critic
from TD3Components.ReplayBuffer import ReplayBuffer

class TD3:
    def __init__(
            self,
         state_dim: int,
         action_dim: int,
         privileged_state_dim: int,
         actor_lr: float = 1e-3,
         critic_lr: float = 1e-3,

         gamma: float = 0.99,
         tau: float = 0.005,

         policy_noise: float = 0.2,
         noise_clip: float = 0.5,

         policy_delay: int = 2,

         replay_capacity: int = 1000000,
         batch_size: int = 256,

         exploration_noise: float = 0.1,

         device: str | None = None,
     ):
        self.action_dim = action_dim
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gamma = gamma
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay
        self.batch_size = batch_size

        self.exploration_noise = exploration_noise


        self.actor = Actor(
            state_dim = state_dim,
            action_dim = action_dim,
        )

        self.actor_target = copy.deepcopy(self.actor).to(self.device)


        #twin critic
        self.critic = Critic(
            actor_action_dim=action_dim,
            actor_state_dim=state_dim,
            privilege_state_dim= privileged_state_dim
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.training_step = 0

    def select_action(
        self,
        estimated_state: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        state = torch.as_tensor(
            estimated_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)
        action = action.cpu().numpy()[0]
        if explore:
            action += np.random.normal(
                0.0,
                self.exploration_noise,
                size = action.shape
            )
        return np.clip(action, -1.0, 1.0)
    # store transition

    def store_transition(
            self,
            estimated_state,
            privileged_state,
            action,
            reward,
            next_estimated_state,
            next_privileged_state,
            done
    ):
        self.replay_buffer.add(
            estimated_state,
            privileged_state,
            action,
            reward,
            next_estimated_state,
            next_privileged_state,
            done
        )

    #training
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)

        (estimated_state,
         privileged_state,
         action,
         reward,
         next_estimated_state,
         next_privileged_state,
         done) = self._batch_to_tensors(batch)


        with torch.no_grad():
            next_action = self.actor_target(next_estimated_state)
            noise = torch.randn_like(next_action) * self.policy_noise

            noise = torch.clamp(
                noise,
                -self.noise_clip,
                self.noise_clip
            )
            next_action = next_action + noise

            next_action = torch.clamp(
                next_action,
                -1.0,
                1.0
            )

            target_q1, target_q2 = self.critic_target(
                next_action,
                next_estimated_state,
                next_privileged_state
            )

            target_q = torch.min(target_q1, target_q2)

            target_q = reward + (self.gamma * (1.0 - done) * target_q)



        current_q1, current_q2 = self.critic(
            action,
            estimated_state,
            privileged_state
        )

        critic_loss = (
            F.mse_loss(current_q1, target_q)
            +
            F.mse_loss(current_q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm = 10.0)
        self.critic_optimizer.step()

        self.training_step += 1


        actor_loss = None
        if self.training_step % self.policy_delay == 0:
            actor_action = self.actor(
                estimated_state
            )
            actor_loss = -self.critic.q1_forward(
                actor_action,
                estimated_state,
                privileged_state
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm = 10.0
            )
            self.actor_optimizer.step()


            self._soft_update(
                self.actor,
                self.actor_target,
            )
            self._soft_update(
                self.critic,
                self.critic_target,
            )
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":(
                actor_loss.item()
                if actor_loss is not None
                else None
            ),
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item(),
        }
    def _batch_to_tensors(self,batch):
        estimated_state = torch.as_tensor(
            np.stack([
                t.estimated_state
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        )
        privileged_state = torch.as_tensor(
            np.stack([
                t.privileged_state
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        )
        action = torch.as_tensor(
            np.stack([
                t.action
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        )

        reward = torch.as_tensor(
            np.array([
                t.reward
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1,1)
        next_estimated_state = torch.as_tensor(
            np.stack([
                t.next_estimated_state
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        )
        next_privileged_state = torch.as_tensor(
            np.stack([
                t.next_privileged_state
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(
            np.array([
                t.done
                for t in batch
            ]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1,1)
        return(
            estimated_state,
            privileged_state,
            action,
            reward,
            next_estimated_state,
            next_privileged_state,
            done
        )
    def _soft_update(
        self,
        source,
        target,
    ):
        for source_param, target_param in zip(
            source.parameters(),
            target.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data
                +
                (1.0 - self.tau) * target_param.data
            )




