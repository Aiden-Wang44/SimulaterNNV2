import numpy as np

from Env import RobotEnv
from TD3Components.TD3 import TD3
from Visualizer.visualizer import Visualizer

def train(
    env: RobotEnv,
    agent: TD3,
    episodes: int = 500,
    warmup_steps: int = 2_000,
    train_after: int = 1_000,
    evaluate_every: int = 25,
    evaluation_episodes: int = 5,
):
    total_steps = 0

    episode_rewards = []

    for episode in range(episodes):
        actor_state = env.reset()

        # The critic needs the privileged state corresponding to s_t.
        privileged_state = env.get_privileged_state()

        episode_reward = 0.0
        episode_steps = 0

        done = False

        while not done:
            # ---------------------------------------------------------
            # Select action
            # ---------------------------------------------------------
            if total_steps < warmup_steps:
                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=agent.action_dim,
                ).astype(np.float32)
            else:
                action = agent.select_action(
                    actor_state,
                    explore=True,
                )

            # ---------------------------------------------------------
            # Step environment
            # ---------------------------------------------------------
            (
                next_actor_state,
                next_privileged_state,
                reward,
                done,
                info,
            ) = env.step(action)

            # ---------------------------------------------------------
            # Store transition
            #
            # This is:
            #
            # s_t, p_t, a_t, r_t, s_{t+1}, p_{t+1}
            # ---------------------------------------------------------
            agent.store_transition(
                estimated_state=actor_state,
                privileged_state=privileged_state,
                action=action,
                reward=reward,
                next_estimated_state=next_actor_state,
                next_privileged_state=next_privileged_state,
                done=done,
            )

            # Move to next state.
            actor_state = next_actor_state
            privileged_state = next_privileged_state

            episode_reward += reward
            episode_steps += 1
            total_steps += 1
#Train TD3
            if total_steps >= train_after:
                agent.train()

        episode_rewards.append(episode_reward)

        average_reward = np.mean(
            episode_rewards[-25:]
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Steps {episode_steps:4d} | "
            f"Reward {episode_reward:9.3f} | "
            f"Avg25 {average_reward:9.3f} | "
            f"Total Steps {total_steps:7d}"
        )
#Periodic Evaluation
        if (
            (episode + 1) % evaluate_every == 0
        ):
            evaluate(
                env=env,
                agent=agent,
                episodes=evaluation_episodes,
            )

    return episode_rewards

def visualize_run(env, agent, visualizer_settings):
    actor_state = env.reset()

# Create a fresh visualizer using the reset environment.
    visualizer = Visualizer(
        env.robot,
        env.path_one,
        visualizer_settings,
        env.time_step,
    )

    done = False

    while not done:
        action = agent.select_action(
            actor_state,
            explore=False,
        )

        (
            actor_state,
            _,
            reward,
            done,
            info,
        ) = env.step(action)

        visualizer.draw()

    visualizer.show()
def evaluate(
    env: RobotEnv,
    agent: TD3,
    episodes: int = 5,
):
    rewards = []
    final_position_errors = []
    final_heading_errors = []

    for _ in range(episodes):
        actor_state = env.reset()

        done = False
        episode_reward = 0.0
        last_info = None

        while not done:
            action = agent.select_action(
                actor_state,
                explore=False,
            )

            (
                actor_state,
                _,
                reward,
                done,
                info,
            ) = env.step(action)

            episode_reward += reward
            last_info = info

        rewards.append(episode_reward)

        if last_info is not None:
            final_position_errors.append(
                last_info["position_error"]
            )
            final_heading_errors.append(
                last_info["heading_error"]
            )

    print(
        f"  Evaluation | "
        f"Reward {np.mean(rewards):9.3f} | "
        f"Position Error "
        f"{np.mean(final_position_errors):7.3f} in | "
        f"Heading Error "
        f"{np.degrees(np.mean(final_heading_errors)):7.3f} deg"
    )

    return {
        "reward": float(np.mean(rewards)),
        "position_error": float(
            np.mean(final_position_errors)
        ),
        "heading_error": float(
            np.mean(final_heading_errors)
        ),
    }