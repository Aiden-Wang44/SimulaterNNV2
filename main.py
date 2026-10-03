import random

import numpy as np
import torch

from Env import RobotEnv
from TD3Components.TD3 import TD3
from train import train, visualize_run
from Visualizer.visualizer import Settings

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    seed = 42
    set_seed(seed)

    env = RobotEnv()

    agent = TD3(
        state_dim=4,
        action_dim=3,
        privileged_state_dim=6,


        replay_capacity=5_000,
        batch_size=32,

        gamma=0.99,
        tau=0.005,

        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,

        exploration_noise=0.1,

        actor_lr=1e-3,
        critic_lr=1e-3,
    )

    train(
        env=env,
        agent=agent,

        episodes=10,

        warmup_steps=100,
        train_after=100,

        evaluate_every=5,
        evaluation_episodes=2,
    )
    settings = Settings()
    visualize_run(
        env=env,
        agent=agent,visualizer_settings=settings )



if __name__ == "__main__":
    main()
